using System.IO;
using OpenCvSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using TsrApp.Models;

// Size/Point exist in both OpenCvSharp and ImageSharp; here they always mean the
// OpenCvSharp ones (drawing on Mat). ImageSharp is only used via Image<Rgb24>.
using Size = OpenCvSharp.Size;
using Point = OpenCvSharp.Point;

namespace TsrApp.Services;

/// <summary>Progress of an export run: <paramref name="Current"/> frames written of <paramref name="Total"/>.</summary>
public readonly record struct ExportProgress(int Current, int Total);

/// <summary>
/// Offline renderer: re-reads a video frame by frame (every frame, no real-time
/// pacing), runs each through the detection→classification→tracking pipeline, and
/// burns the track boxes + labels into the frame with OpenCV before writing to a
/// new mp4 at the source fps/resolution. Uses its own SignTracker (not the live
/// one) and does NOT log tracks — this is a render, not an observation.
/// </summary>
public sealed class VideoExportService
{
    private const double FallbackFps = 25.0;

    // BGR colors matching ConfidenceToColorConverter (>0.9 green, >0.7 yellow, else red).
    private static readonly Scalar Green = new(50, 125, 46);    // #2E7D32
    private static readonly Scalar Yellow = new(37, 168, 249);  // #F9A825
    private static readonly Scalar Red = new(40, 40, 198);      // #C62828

    private readonly PipelineService _pipeline;

    public VideoExportService(PipelineService pipeline) => _pipeline = pipeline;

    public void Export(string inputPath, string outputPath, IProgress<ExportProgress>? progress, CancellationToken token)
    {
        VideoCapture? capture = null;
        VideoWriter? writer = null;
        using var frame = new Mat();
        bool completed = false;

        try
        {
            capture = new VideoCapture(inputPath);
            if (!capture.IsOpened())
                throw new InvalidOperationException($"Не удалось открыть видео: {inputPath}");

            double fps = capture.Get(VideoCaptureProperties.Fps);
            if (double.IsNaN(fps) || fps <= 0) fps = FallbackFps;
            int width = (int)capture.Get(VideoCaptureProperties.FrameWidth);
            int height = (int)capture.Get(VideoCaptureProperties.FrameHeight);
            double frameCount = capture.Get(VideoCaptureProperties.FrameCount);
            int total = (double.IsNaN(frameCount) || frameCount < 0) ? 0 : (int)frameCount;

            writer = OpenWriter(outputPath, fps, new Size(width, height));
            if (writer is null)
                throw new InvalidOperationException("Не удалось создать VideoWriter (нет доступного кодека).");

            var tracker = new SignTracker();
            int index = 0;
            while (!token.IsCancellationRequested && capture.Read(frame) && !frame.Empty())
            {
                // Offline time from frame index + fps, NOT wall clock, so the
                // tracker's stale-track threshold behaves correctly.
                long nowMs = (long)(index * 1000.0 / fps);

                IReadOnlyList<SignTrack> active;
                using (Image<Rgb24> rgb = MatImageConverter.ToImageSharpRgb24(frame))
                {
                    IReadOnlyList<DetectedSign> signs = _pipeline.Process(rgb);
                    active = tracker.Update(signs, index, nowMs).Active;
                }

                DrawOverlay(frame, active);
                writer.Write(frame);

                index++;
                progress?.Report(new ExportProgress(index, total));
            }

            completed = !token.IsCancellationRequested;
        }
        finally
        {
            // Dispose the writer before any delete so the file handle is released.
            writer?.Dispose();
            capture?.Dispose();

            if (!completed && File.Exists(outputPath))
            {
                try { File.Delete(outputPath); } catch { /* best effort: don't mask the real outcome */ }
            }
        }

        token.ThrowIfCancellationRequested();
    }

    /// <summary>Prefer H.264 (avc1); fall back to mp4v, which the Windows runtime reliably ships.</summary>
    private static VideoWriter? OpenWriter(string path, double fps, Size size)
    {
        foreach (string codec in new[] { "avc1", "mp4v" })
        {
            var writer = new VideoWriter(path, VideoWriter.FourCC(codec), fps, size);
            if (writer.IsOpened())
                return writer;
            writer.Dispose();
        }
        return null;
    }

    private static void DrawOverlay(Mat frame, IReadOnlyList<SignTrack> tracks)
    {
        foreach (SignTrack t in tracks)
        {
            Scalar color = t.Confidence > 0.9f ? Green : t.Confidence > 0.7f ? Yellow : Red;

            int x = (int)Math.Round(t.Box.X);
            int y = (int)Math.Round(t.Box.Y);
            int w = (int)Math.Round(t.Box.Width);
            int h = (int)Math.Round(t.Box.Height);
            Cv2.Rectangle(frame, new Rect(x, y, w, h), color, thickness: 2);

            string label = $"{t.ClassName} {t.Confidence:P0} #{t.Id}";
            const HersheyFonts font = HersheyFonts.HersheySimplex;
            const double scale = 0.5;
            const int textThickness = 1;
            Size textSize = Cv2.GetTextSize(label, font, scale, textThickness, out int baseline);

            // Label band just above the box (clamped so it stays on-screen).
            int labelTop = Math.Max(0, y - textSize.Height - baseline - 4);
            Cv2.Rectangle(
                frame,
                new Rect(x, labelTop, textSize.Width + 6, textSize.Height + baseline + 4),
                color,
                thickness: -1);
            Cv2.PutText(
                frame, label,
                new Point(x + 3, labelTop + textSize.Height + 2),
                font, scale, Scalar.White, textThickness, LineTypes.AntiAlias);
        }
    }
}
