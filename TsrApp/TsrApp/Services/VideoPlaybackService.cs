using System.Diagnostics;
using System.IO;
using OpenCvSharp;

namespace TsrApp.Services;

/// <summary>
/// Carries a decoded frame to <see cref="VideoPlaybackService.FrameReady"/>
/// subscribers. See the event's documentation for the strict lifetime rules of
/// <see cref="Frame"/>.
/// </summary>
public sealed class VideoFrameEventArgs : EventArgs
{
    public VideoFrameEventArgs(Mat frame, int frameIndex)
    {
        Frame = frame;
        FrameIndex = frameIndex;
    }

    /// <summary>The decoded BGR8 frame. Owned by the service — do not dispose it.</summary>
    public Mat Frame { get; }

    /// <summary>Zero-based index of this frame within the file.</summary>
    public int FrameIndex { get; }
}

/// <summary>
/// Decodes an mp4 file with OpenCV's <see cref="VideoCapture"/> on a dedicated
/// background thread, pacing frame delivery to the file's own FPS, and pushes
/// each frame out via <see cref="FrameReady"/>. The UI thread is never blocked.
///
/// Frame ownership: the service is the sole owner of the <see cref="Mat"/> it
/// reads into and is the only one that disposes it. Subscribers must not dispose
/// the frame they receive. See <see cref="FrameReady"/> for how long the frame
/// stays valid.
/// </summary>
public sealed class VideoPlaybackService : IDisposable
{
    private const double FallbackFps = 25.0;

    private readonly Mat _frame = new();
    private readonly ManualResetEventSlim _resumeGate = new(initialState: false);
    private readonly object _sync = new();

    private VideoCapture? _capture;
    private Thread? _worker;
    private CancellationTokenSource? _cts;
    private volatile int _currentFrameIndex = -1;
    private bool _disposed;

    /// <summary>Frames per second reported by the file (falls back to 25 if unknown).</summary>
    public double Fps { get; private set; }

    /// <summary>Total number of frames reported by the file (0 if unknown).</summary>
    public int TotalFrames { get; private set; }

    /// <summary>Zero-based index of the most recently delivered frame (-1 before playback).</summary>
    public int CurrentFrameIndex => _currentFrameIndex;

    /// <summary>
    /// Raised on the background decode thread for every decoded frame.
    /// <para>
    /// The <see cref="VideoFrameEventArgs.Frame"/> is owned by the service and is
    /// only guaranteed valid for the duration of the handler call — the service
    /// reuses and eventually disposes that same <see cref="Mat"/>. Therefore:
    /// </para>
    /// <list type="bullet">
    /// <item>UI subscribers must marshal to the UI thread with
    /// <c>Dispatcher.BeginInvoke</c> (asynchronous) — never the synchronous
    /// <c>Invoke</c> — to avoid deadlocking against <see cref="Stop"/>/<see cref="Dispose"/>,
    /// which join this thread. Copy out what you need first (e.g. via
    /// <see cref="MatImageConverter.ToBitmapSource"/>, which produces an
    /// independent frozen bitmap).</item>
    /// <item>Any subscriber that needs the frame for longer than the handler call
    /// (e.g. a future inference thread) must take its own <see cref="Mat.Clone"/>
    /// and is then responsible for disposing that clone.</item>
    /// </list>
    /// </summary>
    public event EventHandler<VideoFrameEventArgs>? FrameReady;

    /// <summary>Raised on the background thread when the end of the file is reached.</summary>
    public event EventHandler? PlaybackCompleted;

    /// <summary>
    /// Opens an mp4 file for playback. Any previous playback is stopped first, so
    /// the same service instance can be reused across files. Playback does not
    /// start until <see cref="Play"/> is called.
    /// </summary>
    public void Open(string path)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path is empty.", nameof(path));
        if (!File.Exists(path))
            throw new FileNotFoundException("Video file not found.", path);

        Stop();

        var capture = new VideoCapture(path);
        if (!capture.IsOpened())
        {
            capture.Dispose();
            throw new InvalidOperationException($"Failed to open video: {path}");
        }

        double fps = capture.Get(VideoCaptureProperties.Fps);
        Fps = (double.IsNaN(fps) || fps <= 0) ? FallbackFps : fps;

        double frames = capture.Get(VideoCaptureProperties.FrameCount);
        TotalFrames = (double.IsNaN(frames) || frames < 0) ? 0 : (int)frames;

        _capture = capture;
        _currentFrameIndex = -1;
    }

    /// <summary>
    /// Starts playback, or resumes it after <see cref="Pause"/>. Starting the
    /// worker thread and resuming are both idempotent-safe to call.
    /// </summary>
    public void Play()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_capture is null)
            throw new InvalidOperationException("Open a video before calling Play.");

        lock (_sync)
        {
            if (_worker is null || !_worker.IsAlive)
            {
                _cts = new CancellationTokenSource();
                _resumeGate.Set();
                _worker = new Thread(() => RunDecodeLoop(_cts.Token))
                {
                    IsBackground = true,
                    Name = "VideoPlaybackDecode",
                };
                _worker.Start();
            }
            else
            {
                _resumeGate.Set(); // resume a paused loop
            }
        }
    }

    /// <summary>Pauses delivery without releasing the file. Resume with <see cref="Play"/>.</summary>
    public void Pause()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _resumeGate.Reset();
    }

    /// <summary>
    /// Stops playback, joins the decode thread and releases the file. Safe to call
    /// when nothing is playing. The service can be reopened afterwards.
    /// </summary>
    public void Stop()
    {
        Thread? worker;
        CancellationTokenSource? cts;
        lock (_sync)
        {
            worker = _worker;
            cts = _cts;
            _worker = null;
            _cts = null;
        }

        // Cancel, then open the gate so a paused loop unblocks and observes it.
        cts?.Cancel();
        _resumeGate.Set();

        // Only join from a different thread; the worker never calls Stop on itself.
        if (worker is not null && worker.IsAlive && worker != Thread.CurrentThread)
            worker.Join();

        cts?.Dispose();
        _resumeGate.Reset();

        _capture?.Dispose();
        _capture = null;
        _currentFrameIndex = -1;
    }

    private void RunDecodeLoop(CancellationToken token)
    {
        VideoCapture? capture = _capture;
        if (capture is null)
            return;

        double frameInterval = 1000.0 / Fps; // milliseconds per frame
        var stopwatch = new Stopwatch();

        try
        {
            while (!token.IsCancellationRequested)
            {
                // Block here while paused; throws if cancelled during the wait.
                _resumeGate.Wait(token);

                stopwatch.Restart();

                if (!capture.Read(_frame) || _frame.Empty())
                {
                    PlaybackCompleted?.Invoke(this, EventArgs.Empty);
                    return;
                }

                int index = _currentFrameIndex + 1;
                _currentFrameIndex = index;

                // Synchronous: the handler must finish using the frame before we
                // loop and overwrite it on the next Read (see FrameReady docs).
                FrameReady?.Invoke(this, new VideoFrameEventArgs(_frame, index));

                // Pace to the file's FPS; wake immediately if cancelled.
                double remaining = frameInterval - stopwatch.Elapsed.TotalMilliseconds;
                if (remaining > 0 && token.WaitHandle.WaitOne((int)remaining))
                    return; // cancellation signalled during the wait
            }
        }
        catch (OperationCanceledException)
        {
            // Normal Stop/Dispose path while paused — fall through to exit.
        }
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;

        Stop();
        _frame.Dispose();
        _resumeGate.Dispose();
    }
}
