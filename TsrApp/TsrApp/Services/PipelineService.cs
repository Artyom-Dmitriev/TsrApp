using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using TsrApp.Models;

namespace TsrApp.Services;

/// <summary>
/// Two-stage orchestrator: YOLO detector finds sign boxes, then each box is
/// cropped (padded + squared) and handed to the ResNet-18 classifier.
/// Does not own the services it is given (the ViewModel disposes them).
/// </summary>
public sealed class PipelineService
{
    private readonly DetectorService _detector;
    private readonly ClassifierService _classifier;

    public PipelineService(DetectorService detector, ClassifierService classifier)
    {
        _detector = detector;
        _classifier = classifier;
    }

    public IReadOnlyList<DetectedSign> Process(Image<Rgb24> frame, float pad = 0.15f)
    {
        IReadOnlyList<DetectorBox> boxes = _detector.Detect(frame);
        if (boxes.Count == 0)
            return Array.Empty<DetectedSign>(); // no signs found is a normal case

        var signs = new List<DetectedSign>(boxes.Count);
        foreach (DetectorBox box in boxes)
        {
            Rectangle rect = BuildSquareCrop(box, frame.Width, frame.Height, pad);
            using Image<Rgb24> crop = frame.Clone(c => c.Crop(rect));
            PredictionResult cls = _classifier.Predict(crop);
            signs.Add(new DetectedSign(box, cls.ClassId, cls.ClassName, cls.Confidence, box.Score));
        }

        return signs;
    }

    /// <summary>
    /// Expand the box by <paramref name="pad"/> (fraction of each side), make it
    /// square around the same center (extend the shorter side symmetrically),
    /// then clamp to the frame. The classifier stretches its input to 224x224,
    /// so a square crop avoids aspect distortion and the padding adds context.
    /// </summary>
    private static Rectangle BuildSquareCrop(DetectorBox box, int frameW, int frameH, float pad)
    {
        float cx = box.X + box.Width / 2f;
        float cy = box.Y + box.Height / 2f;

        float paddedW = box.Width * (1f + 2f * pad);
        float paddedH = box.Height * (1f + 2f * pad);
        float side = Math.Max(paddedW, paddedH);

        float x1 = cx - side / 2f;
        float y1 = cy - side / 2f;
        float x2 = cx + side / 2f;
        float y2 = cy + side / 2f;

        int left = (int)Math.Round(Math.Clamp(x1, 0f, frameW));
        int top = (int)Math.Round(Math.Clamp(y1, 0f, frameH));
        int right = (int)Math.Round(Math.Clamp(x2, 0f, frameW));
        int bottom = (int)Math.Round(Math.Clamp(y2, 0f, frameH));

        // Guarantee a non-degenerate crop that stays inside the frame.
        int width = Math.Max(1, right - left);
        int height = Math.Max(1, bottom - top);
        if (left + width > frameW) left = frameW - width;
        if (top + height > frameH) top = frameH - height;

        return new Rectangle(left, top, width, height);
    }
}
