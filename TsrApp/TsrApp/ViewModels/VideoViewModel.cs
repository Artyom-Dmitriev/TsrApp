using System.Collections.Concurrent;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using OpenCvSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using TsrApp.Models;
using TsrApp.Services;

namespace TsrApp.ViewModels;

/// <summary>
/// Owns a <see cref="VideoPlaybackService"/> and drives the "Видео" tab: opening
/// an mp4, play/pause/stop, pushing decoded frames to the UI, and (optionally)
/// running the two-stage detection+classification pipeline on a dedicated
/// background thread with a latest-frame-wins policy. Inference results are fed
/// through an IoU tracker so boxes are stable across results; CSV logging of the
/// closed tracks is added in the next step.
/// </summary>
public partial class VideoViewModel : ObservableObject, IDisposable
{
    private const int StatusUpdateIntervalMs = 250; // ~4 status refreshes/sec

    private readonly VideoPlaybackService _service;
    private readonly PipelineService _pipeline; // owned by MainViewModel, not disposed here
    private readonly Dispatcher _dispatcher = Dispatcher.CurrentDispatcher;
    private readonly Stopwatch _statusThrottle = Stopwatch.StartNew();

    // Coalescing buffers: producers (decode thread) drop their work here; the UI
    // pulls the most recent value. Lets slow rendering silently skip frames
    // instead of growing the Dispatcher queue.
    private BitmapSource? _latestFrame;
    private int _renderPending;
    private volatile int _latestFrameIndex = -1;
    private int _statusPending;

    // --- Inference (latest-frame-wins) ---
    // Single-slot Mat clone handed from the decode thread to the inference thread.
    // Whoever pulls a clone out of the slot (eviction in FrameReady, or the
    // inference thread on take) owns disposing it.
    private readonly object _inferenceSync = new();
    private readonly ManualResetEventSlim _frameSignal = new(initialState: false);
    private readonly Stopwatch _inferenceStopwatch = new();
    private PendingFrame? _pendingFrame;
    private Thread? _inferenceThread;
    private CancellationTokenSource? _inferenceCts;
    private int _inferenceFrames;
    // Tags results with the session they were computed in, so stale results that
    // were already queued before a Stop/file-change are ignored by the UI.
    private int _inferenceGeneration;

    // Tracker is created per session and touched only by the inference thread
    // (created before the thread starts, reset after it joins) — no locking.
    private SignTracker? _tracker;

    // Summaries of tracks that have closed (naturally or on reset). Concurrent
    // because both the inference thread and the UI thread (on reset) enqueue.
    // Drained on the UI thread, where the logger and history live.
    private readonly ConcurrentQueue<TrackSummary> _closedTracks = new();
    private int _drainPending;

    private InferenceResult? _latestResult;
    private int _resultPending;

    // --- Pipeline timing instrumentation (inference thread only) ---
    private const int TimingWindow = 50; // rolling average over ~50 processed frames
    private readonly RollingAverage _convAvg = new(TimingWindow);
    private readonly RollingAverage _detAvg = new(TimingWindow);
    private readonly RollingAverage _clsAvg = new(TimingWindow);
    private readonly RollingAverage _clsCountAvg = new(TimingWindow);
    private readonly RollingAverage _totalAvg = new(TimingWindow);
    private string _timingText = ""; // set on the UI thread, read in UpdateStatus

    // --- Offline export ---
    private readonly VideoExportService _exporter;
    private CancellationTokenSource? _exportCts;
    private Task? _exportTask;

    private string? _currentPath;
    private string _stateNote = "";
    private bool _disposed;

    /// <summary>
    /// Overlay boxes for the live tracks. Mutable item-VMs so their positions can
    /// be re-extrapolated every rendered frame without rebuilding the collection.
    /// </summary>
    public ObservableCollection<TrackBoxViewModel> VideoTracks { get; } = new();

    /// <summary>
    /// Raised on the UI thread once per closed track, with a ready-to-log entry.
    /// MainViewModel persists it and adds it to the history. One event per track —
    /// never per frame or per inference result.
    /// </summary>
    public event Action<PredictionLogEntry>? EntryLogged;

    /// <summary>A single inference frame in flight: the clone plus its frame index.</summary>
    private sealed record PendingFrame(Mat Mat, int Index);

    [ObservableProperty] private ImageSource? _frameSource;
    [ObservableProperty] private int _frameWidth;
    [ObservableProperty] private int _frameHeight;

    [ObservableProperty] private string _fileName = "";
    [ObservableProperty] private double _fps;
    [ObservableProperty] private int _totalFrames;
    [ObservableProperty] private int _currentFrameIndex = -1;
    [ObservableProperty] private double _inferenceFps;
    [ObservableProperty] private string _statusText = "";

    [ObservableProperty] private bool _isInferenceEnabled;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(ExportWithOverlayCommand))]
    [NotifyCanExecuteChangedFor(nameof(CancelExportCommand))]
    [NotifyCanExecuteChangedFor(nameof(PlayCommand))]
    [NotifyCanExecuteChangedFor(nameof(OpenVideoCommand))]
    private bool _isExporting;

    [ObservableProperty] private int _exportCurrent;
    [ObservableProperty] private int _exportTotal;

    private sealed record InferenceResult(int Generation, IReadOnlyList<SignTrack> Tracks);

    /// <summary>Fixed-window moving average over a circular buffer (single-threaded use).</summary>
    private sealed class RollingAverage
    {
        private readonly double[] _buf;
        private int _index;
        private int _count;
        private double _sum;

        public RollingAverage(int capacity) => _buf = new double[capacity];

        public void Add(double value)
        {
            if (_count == _buf.Length)
                _sum -= _buf[_index];
            else
                _count++;
            _buf[_index] = value;
            _sum += value;
            _index = (_index + 1) % _buf.Length;
        }

        public double Average => _count == 0 ? 0 : _sum / _count;

        public void Reset()
        {
            Array.Clear(_buf);
            _index = 0;
            _count = 0;
            _sum = 0;
        }
    }

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(PlayCommand))]
    [NotifyCanExecuteChangedFor(nameof(PauseCommand))]
    [NotifyCanExecuteChangedFor(nameof(StopCommand))]
    [NotifyCanExecuteChangedFor(nameof(ExportWithOverlayCommand))]
    private bool _isFileOpen;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(PlayCommand))]
    [NotifyCanExecuteChangedFor(nameof(PauseCommand))]
    [NotifyCanExecuteChangedFor(nameof(StopCommand))]
    [NotifyCanExecuteChangedFor(nameof(ExportWithOverlayCommand))]
    private bool _isPlaying;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(PlayCommand))]
    [NotifyCanExecuteChangedFor(nameof(PauseCommand))]
    [NotifyCanExecuteChangedFor(nameof(StopCommand))]
    private bool _isPaused;

    public VideoViewModel(PipelineService pipeline)
    {
        _pipeline = pipeline;
        _exporter = new VideoExportService(pipeline);
        _service = new VideoPlaybackService();
        _service.FrameReady += OnFrameReady;
        _service.PlaybackCompleted += OnPlaybackCompleted;
        UpdateStatus();
    }

    [RelayCommand(CanExecute = nameof(CanOpenVideo))]
    private void OpenVideo()
    {
        var dlg = new OpenFileDialog
        {
            Title = "Выберите видео",
            Filter = "Видео (*.mp4;*.webm;*.avi;*.mkv;*.mov)|*.mp4;*.webm;*.avi;*.mkv;*.mov"
                   + "|Все файлы (*.*)|*.*",
        };
        if (dlg.ShowDialog() != true) return;

        // Stop any inference running on the previous file before swapping.
        StopInference();
        VideoTracks.Clear();

        // Service.Open() stops any current playback internally, so swapping the
        // file mid-playback is safe.
        _currentPath = dlg.FileName;
        _service.Open(_currentPath);

        FileName = Path.GetFileName(_currentPath);
        Fps = _service.Fps;
        TotalFrames = _service.TotalFrames;
        CurrentFrameIndex = -1;
        _latestFrameIndex = -1;
        FrameSource = null;
        FrameWidth = 0;
        FrameHeight = 0;

        IsFileOpen = true;
        IsPlaying = false;
        IsPaused = false;
        _stateNote = "";
        _statusThrottle.Restart();
        UpdateStatus();
    }

    private bool CanOpenVideo() => !IsExporting;

    [RelayCommand(CanExecute = nameof(CanPlay))]
    private void Play()
    {
        if (IsInferenceEnabled) StartInference();
        _service.Play();
        IsPlaying = true;
        IsPaused = false;
        _stateNote = "воспроизведение";
        UpdateStatus();
    }

    private bool CanPlay() => IsFileOpen && !IsPlaying && !IsExporting;

    [RelayCommand(CanExecute = nameof(CanPause))]
    private void Pause()
    {
        _service.Pause();
        IsPlaying = false;
        IsPaused = true;
        _stateNote = "пауза";
        UpdateStatus();
    }

    private bool CanPause() => IsPlaying;

    [RelayCommand(CanExecute = nameof(CanStop))]
    private void Stop()
    {
        StopInference();
        VideoTracks.Clear();
        _service.Stop();
        ReloadForReplay(); // rewind to frame 0 so Play restarts from the beginning
        IsPlaying = false;
        IsPaused = false;
        CurrentFrameIndex = -1;
        _stateNote = "остановлено";
        UpdateStatus();
    }

    private bool CanStop() => IsFileOpen && (IsPlaying || IsPaused);

    /// <summary>Toggle handler: spin the inference loop up/down on the fly.</summary>
    partial void OnIsInferenceEnabledChanged(bool value)
    {
        if (value)
        {
            // Only meaningful while playing; if idle the loop just waits for frames.
            if (IsPlaying) StartInference();
        }
        else
        {
            StopInference();
            VideoTracks.Clear();
        }
        UpdateStatus();
    }

    /// <summary>Re-open the current file so a finished/stopped clip can be replayed.</summary>
    private void ReloadForReplay()
    {
        if (_currentPath is null) return;
        _service.Open(_currentPath);
        CurrentFrameIndex = -1;
        _latestFrameIndex = -1;
    }

    [RelayCommand(CanExecute = nameof(CanExport))]
    private async Task ExportWithOverlayAsync()
    {
        if (_currentPath is null) return;

        var dlg = new SaveFileDialog
        {
            Title = "Экспорт видео с разметкой",
            Filter = "Видео MP4 (*.mp4)|*.mp4",
            FileName = Path.GetFileNameWithoutExtension(_currentPath) + "_annotated.mp4",
            DefaultExt = ".mp4",
        };
        if (dlg.ShowDialog() != true) return;

        string input = _currentPath;
        string output = dlg.FileName;
        _exportCts = new CancellationTokenSource();
        CancellationToken token = _exportCts.Token;

        IsExporting = true;
        ExportCurrent = 0;
        ExportTotal = TotalFrames;
        _stateNote = "экспорт…";
        UpdateStatus();

        var progress = new Progress<ExportProgress>(p =>
        {
            ExportCurrent = p.Current;
            if (p.Total > 0) ExportTotal = p.Total;
            UpdateStatus();
        });

        try
        {
            _exportTask = Task.Run(() => _exporter.Export(input, output, progress, token), token);
            await _exportTask;
            _stateNote = "экспорт завершён";
        }
        catch (OperationCanceledException)
        {
            _stateNote = "экспорт отменён";
        }
        catch (Exception ex)
        {
            _stateNote = "ошибка экспорта";
            System.Windows.MessageBox.Show($"Не удалось экспортировать видео:\n{ex.Message}",
                "Ошибка экспорта", System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Error);
        }
        finally
        {
            _exportTask = null;
            _exportCts.Dispose();
            _exportCts = null;
            IsExporting = false;
            UpdateStatus();
        }
    }

    private bool CanExport() => IsFileOpen && !IsPlaying && !IsExporting;

    [RelayCommand(CanExecute = nameof(CanCancelExport))]
    private void CancelExport() => _exportCts?.Cancel();

    private bool CanCancelExport() => IsExporting;

    /// <summary>
    /// Runs on the decode thread. Per the <see cref="VideoPlaybackService.FrameReady"/>
    /// contract: convert synchronously into an independent frozen bitmap and never
    /// keep the Mat. The bitmap is then coalesced — only the latest one is rendered.
    /// </summary>
    private void OnFrameReady(object? sender, VideoFrameEventArgs e)
    {
        if (_disposed) return;

        BitmapSource bmp;
        try
        {
            bmp = MatImageConverter.ToBitmapSource(e.Frame);
        }
        catch
        {
            return; // a single bad frame must not tear down playback
        }

        // Drop the freshest frame into the buffer; post a render only if one
        // isn't already queued, so the Dispatcher queue can't pile up.
        Interlocked.Exchange(ref _latestFrame, bmp);
        if (Interlocked.CompareExchange(ref _renderPending, 1, 0) == 0)
            _dispatcher.BeginInvoke(RenderLatestFrame);

        // Feed the inference slot (latest-frame-wins). Per the FrameReady contract
        // we only Clone here — the heavy work happens on the inference thread. The
        // evicted clone is disposed by us (the evictor); the consumed clone by the
        // inference thread.
        if (IsInferenceEnabled)
        {
            var holder = new PendingFrame(e.Frame.Clone(), e.FrameIndex);
            PendingFrame? evicted = Interlocked.Exchange(ref _pendingFrame, holder);
            evicted?.Mat.Dispose();
            _frameSignal.Set();
        }

        // Counter is refreshed at most ~4x/sec, also coalesced.
        _latestFrameIndex = e.FrameIndex;
        if (_statusThrottle.ElapsedMilliseconds >= StatusUpdateIntervalMs)
        {
            _statusThrottle.Restart();
            if (Interlocked.CompareExchange(ref _statusPending, 1, 0) == 0)
                _dispatcher.BeginInvoke(RefreshCounter);
        }
    }

    private void RenderLatestFrame()
    {
        // Clear the gate first, then take the newest frame, so a frame that
        // arrives during this call still schedules a follow-up render.
        Interlocked.Exchange(ref _renderPending, 0);
        if (_disposed) return;

        BitmapSource? frame = Interlocked.Exchange(ref _latestFrame, null);
        if (frame is null) return;

        FrameSource = frame;
        if (FrameWidth != frame.PixelWidth) FrameWidth = frame.PixelWidth;
        if (FrameHeight != frame.PixelHeight) FrameHeight = frame.PixelHeight;

        // Re-extrapolate the overlay boxes for this frame.
        UpdateOverlayPositions();
    }

    private void RefreshCounter()
    {
        Interlocked.Exchange(ref _statusPending, 0);
        if (_disposed) return;

        CurrentFrameIndex = _latestFrameIndex;
        UpdateStatus();
    }

    // --- Inference lifecycle (all calls below are on the UI thread) ---

    private void StartInference()
    {
        if (_disposed || !IsInferenceEnabled) return;

        lock (_inferenceSync)
        {
            if (_inferenceThread is { IsAlive: true }) return;

            _inferenceCts = new CancellationTokenSource();
            int generation = ++_inferenceGeneration;
            _inferenceFrames = 0;
            _inferenceStopwatch.Restart();
            _frameSignal.Reset();
            _tracker = new SignTracker(); // fresh tracks per session

            _convAvg.Reset();
            _detAvg.Reset();
            _clsAvg.Reset();
            _clsCountAvg.Reset();
            _totalAvg.Reset();
            _timingText = "";

            CancellationToken token = _inferenceCts.Token;
            _inferenceThread = new Thread(() => RunInferenceLoop(token, generation))
            {
                IsBackground = true,
                Name = "VideoInference",
            };
            _inferenceThread.Start();
        }
    }

    private void StopInference()
    {
        Thread? thread;
        CancellationTokenSource? cts;
        lock (_inferenceSync)
        {
            thread = _inferenceThread;
            cts = _inferenceCts;
            _inferenceThread = null;
            _inferenceCts = null;
        }

        cts?.Cancel();
        _frameSignal.Set(); // unblock the loop's Wait so it can observe cancellation

        if (thread is { IsAlive: true } && thread != Thread.CurrentThread)
            thread.Join();

        cts?.Dispose();
        _frameSignal.Reset();

        // Drain the slot so no live clone is left behind.
        Interlocked.Exchange(ref _pendingFrame, null)?.Mat.Dispose();

        // The thread has joined, so the tracker is no longer in use: close its
        // remaining tracks and hand the summaries off for logging.
        if (_tracker is not null)
        {
            foreach (TrackSummary summary in _tracker.Reset())
                _closedTracks.Enqueue(summary);
            _tracker = null;
        }

        // Flush synchronously while _currentPath still points at this session's
        // file (it changes right after, in OpenVideo).
        DrainClosedTracks();

        // Invalidate any result that was queued before this stop.
        _inferenceGeneration++;
        InferenceFps = 0;
        _timingText = ""; // don't show stale timings after stop
    }

    /// <summary>
    /// UI thread: turn every queued closed-track summary into a log entry and
    /// raise <see cref="EntryLogged"/>. Each summary is dequeued exactly once, so
    /// no track is logged twice.
    /// </summary>
    private void DrainClosedTracks()
    {
        Interlocked.Exchange(ref _drainPending, 0);

        while (_closedTracks.TryDequeue(out TrackSummary? summary))
            EntryLogged?.Invoke(BuildLogEntry(summary));
    }

    private PredictionLogEntry BuildLogEntry(TrackSummary s) => new()
    {
        Timestamp = DateTime.UtcNow.ToString("O"),
        ImagePath = _currentPath ?? "",
        PredictedClassId = s.ClassId,
        PredictedClassName = s.ClassName,
        Confidence = s.Confidence,
        Mode = "Video",
        SourceType = "Track",
        FrameIndex = s.FirstFrame,
        // Same "x,y,w,h" format as the Detector mode, for the last matched box.
        BBox = $"{s.Box.X:0},{s.Box.Y:0},{s.Box.Width:0},{s.Box.Height:0}",
        TrackId = s.Id,
        LastFrame = s.LastConfirmedFrame,
        Confirmations = s.Confirmations,
    };

    /// <summary>
    /// Dedicated inference thread. Waits for a frame, takes the single pending
    /// clone (latest-frame-wins), runs the pipeline, and publishes the result with
    /// a coalesced BeginInvoke. Owns and disposes every clone it consumes.
    /// </summary>
    private void RunInferenceLoop(CancellationToken token, int generation)
    {
        // Warm both sessions before real frames so cold-start cost doesn't show up
        // in the timing stats (the rolling windows were reset in StartInference).
        try { _pipeline.Warmup(); }
        catch { /* warmup failure is non-fatal */ }

        try
        {
            while (!token.IsCancellationRequested)
            {
                _frameSignal.Wait(token);
                _frameSignal.Reset();

                PendingFrame? pending = Interlocked.Exchange(ref _pendingFrame, null);
                if (pending is null)
                    continue; // spurious wake or a Stop signal with an empty slot

                IReadOnlyList<SignTrack> tracks;
                double convMs = 0;
                PipelineTimings pt = default;
                var frameSw = Stopwatch.StartNew(); // (г) full per-frame time
                try
                {
                    IReadOnlyList<DetectedSign> signs;
                    using (Mat frame = pending.Mat)
                    {
                        // (а) Mat -> pipeline input: BGR->RGB + Image creation.
                        var convSw = Stopwatch.StartNew();
                        using Image<Rgb24> rgb = MatImageConverter.ToImageSharpRgb24(frame);
                        convMs = convSw.Elapsed.TotalMilliseconds;

                        // (б) detector + (в) classifier timings come back via pt.
                        signs = _pipeline.Process(rgb, out pt);
                    }

                    // Feed the tracker (only the inference thread touches it).
                    TrackerResult tracked = _tracker!.Update(signs, pending.Index, Environment.TickCount64);
                    tracks = tracked.Active;
                    if (tracked.Closed.Count > 0)
                    {
                        foreach (TrackSummary summary in tracked.Closed)
                            _closedTracks.Enqueue(summary);
                        // Drain on the UI thread (coalesced) so closed tracks are
                        // logged as they happen, without growing the queue.
                        if (Interlocked.CompareExchange(ref _drainPending, 1, 0) == 0)
                            _dispatcher.BeginInvoke(DrainClosedTracks);
                    }
                }
                catch
                {
                    continue; // a single bad frame must not kill the loop (clone already freed by using)
                }

                // Record stage timings into the rolling windows (inference thread only).
                _convAvg.Add(convMs);
                _detAvg.Add(pt.DetectorMs);
                _clsAvg.Add(pt.ClassifierMs);
                _clsCountAvg.Add(pt.ClassifiedCount);
                _totalAvg.Add(frameSw.Elapsed.TotalMilliseconds);

                // Publish (coalesced): only the latest result is drawn.
                Interlocked.Exchange(ref _latestResult, new InferenceResult(generation, tracks));
                if (Interlocked.CompareExchange(ref _resultPending, 1, 0) == 0)
                    _dispatcher.BeginInvoke(ApplyLatestResult);

                // Inference throughput: recompute once per second.
                _inferenceFrames++;
                long ms = _inferenceStopwatch.ElapsedMilliseconds;
                if (ms >= 1000)
                {
                    double fps = _inferenceFrames * 1000.0 / ms;
                    _inferenceFrames = 0;
                    _inferenceStopwatch.Restart();

                    string timing = $"conv {_convAvg.Average:0.0} мс | det {_detAvg.Average:0.0} мс"
                                  + $" | cls {_clsAvg.Average:0.0} мс × {_clsCountAvg.Average:0.#}"
                                  + $" | total {_totalAvg.Average:0.0} мс";

                    _dispatcher.BeginInvoke(() =>
                    {
                        if (_disposed) return;
                        InferenceFps = fps;
                        _timingText = timing;
                        UpdateStatus();
                    });
                }
            }
        }
        catch (OperationCanceledException)
        {
            // Normal Stop/Dispose path — exit quietly.
        }
    }

    private void ApplyLatestResult()
    {
        Interlocked.Exchange(ref _resultPending, 0);
        if (_disposed) return;

        InferenceResult? result = Interlocked.Exchange(ref _latestResult, null);
        if (result is null) return;

        // Ignore results from a previous session (Stop/file-change) or after the
        // toggle was switched off.
        if (result.Generation != _inferenceGeneration || !IsInferenceEnabled) return;

        ReconcileTracks(result.Tracks);
        UpdateOverlayPositions(); // so new/updated boxes appear without waiting a frame
    }

    /// <summary>
    /// Merge the latest active tracks into <see cref="VideoTracks"/> by id —
    /// update existing item-VMs in place, add new ones, drop the ones that are
    /// gone — instead of rebuilding the collection (keeps box identity stable).
    /// </summary>
    private void ReconcileTracks(IReadOnlyList<SignTrack> tracks)
    {
        // Remove boxes whose track is no longer active.
        for (int i = VideoTracks.Count - 1; i >= 0; i--)
        {
            if (!tracks.Any(t => t.Id == VideoTracks[i].Id))
                VideoTracks.RemoveAt(i);
        }

        // Update existing, add new.
        foreach (SignTrack track in tracks)
        {
            TrackBoxViewModel? existing = VideoTracks.FirstOrDefault(v => v.Id == track.Id);
            if (existing is not null)
                existing.Update(track);
            else
                VideoTracks.Add(new TrackBoxViewModel(track));
        }
    }

    /// <summary>Re-extrapolate every overlay box for the current time and frame size.</summary>
    private void UpdateOverlayPositions()
    {
        long now = Environment.TickCount64; // same clock the tracker stamps confirmations with
        int w = FrameWidth, h = FrameHeight;
        foreach (TrackBoxViewModel box in VideoTracks)
            box.UpdatePosition(now, w, h);
    }

    private void OnPlaybackCompleted(object? sender, EventArgs e)
    {
        _dispatcher.BeginInvoke(() =>
        {
            if (_disposed) return;
            IsPlaying = false;
            IsPaused = false;
            _stateNote = "завершено";
            ReloadForReplay();
            UpdateStatus();
        });
    }

    private void UpdateStatus()
    {
        if (!IsFileOpen)
        {
            StatusText = "Файл не выбран";
            return;
        }

        int shown = Math.Max(0, CurrentFrameIndex + 1);
        string note = _stateNote.Length > 0 ? $"    ·    {_stateNote}" : "";
        string infer = IsInferenceEnabled ? $"    Инференс: {InferenceFps:0.0} к/с" : "";
        string timing = (IsInferenceEnabled && _timingText.Length > 0) ? $"    {_timingText}" : "";
        string export = IsExporting ? $"    Экспорт: кадр {ExportCurrent} из {ExportTotal}" : "";
        StatusText = $"{FileName}    FPS: {Fps:0.#}    Кадр: {shown} / {TotalFrames}{infer}{timing}{export}{note}";
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        // Stop an in-flight export: cancel and wait briefly for its teardown to run
        // (it deletes the partial file). The loop checks the token every frame, and
        // Progress.Report is non-blocking, so this returns quickly without deadlock.
        _exportCts?.Cancel();
        try { _exportTask?.Wait(TimeSpan.FromSeconds(5)); }
        catch { /* cancellation / export error already handled by the command */ }

        StopInference();      // joins the inference thread and flushes closed tracks
        DrainClosedTracks();  // flush anything still queued so nothing is lost
        _service.FrameReady -= OnFrameReady;
        _service.PlaybackCompleted -= OnPlaybackCompleted;
        _service.Dispose();
        _frameSignal.Dispose();
    }
}
