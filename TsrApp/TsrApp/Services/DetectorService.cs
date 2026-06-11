using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using TsrApp.Models;

namespace TsrApp.Services;

/// <summary>
/// YOLOv8n single-class traffic-sign detector. This is a 1:1 C# port of the
/// reference Python checker (tsr_train/detector/check_yolo_onnx.py): letterbox
/// to 640x640 -> inference -> decode [1,5,8400] -> undo letterbox -> NMS.
/// </summary>
public sealed class DetectorService : IDisposable
{
    private const int ImgSz = 640;
    // Gray letterbox padding color, identical to check_yolo_onnx.py.
    private static readonly Rgb24 PadColor = new(114, 114, 114);

    private readonly InferenceSession _session;
    private readonly string _inputName;

    public DetectorService(string modelPath)
    {
        // Two sessions (detector + classifier) share one CPU: split the cores and
        // stop idle intra-op threads from spin-waiting so the pools don't fight.
        using var options = new SessionOptions();
        options.IntraOpNumThreads = Math.Max(1, Environment.ProcessorCount / 2);
        options.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
        _session = new InferenceSession(modelPath, options);
        _inputName = _session.InputMetadata.Keys.First();
    }

    /// <summary>Runs one dummy inference so the first real frame isn't cold.</summary>
    public void Warmup()
    {
        using var dummy = new Image<Rgb24>(ImgSz, ImgSz);
        Detect(dummy);
    }

    public IReadOnlyList<DetectorBox> Detect(
        Image<Rgb24> image,
        float confThreshold = 0.25f,
        float iouThreshold = 0.45f)
    {
        int origW = image.Width;
        int origH = image.Height;

        // --- Letterbox (on a clone; the caller's image is untouched) ---
        // scale/round/padding computed exactly as in check_yolo_onnx.py.
        double scale = Math.Min((double)ImgSz / origW, (double)ImgSz / origH);
        int newW = (int)Math.Round(origW * scale, MidpointRounding.ToEven);
        int newH = (int)Math.Round(origH * scale, MidpointRounding.ToEven);
        int padX = (ImgSz - newW) / 2;
        int padY = (ImgSz - newH) / 2;

        float[] input = new float[3 * ImgSz * ImgSz];
        using (Image<Rgb24> resized = image.Clone(ctx => ctx.Resize(newW, newH, KnownResamplers.Triangle)))
        using (Image<Rgb24> canvas = new(ImgSz, ImgSz, PadColor))
        {
            canvas.Mutate(ctx => ctx.DrawImage(resized, new Point(padX, padY), 1f));

            // /255, no ImageNet normalization, RGB order, CHW -> [1,3,640,640].
            int plane = ImgSz * ImgSz;
            canvas.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < ImgSz; y++)
                {
                    Span<Rgb24> row = accessor.GetRowSpan(y);
                    int rowOffset = y * ImgSz;
                    for (int x = 0; x < ImgSz; x++)
                    {
                        Rgb24 p = row[x];
                        int idx = rowOffset + x;
                        input[idx] = p.R / 255f;
                        input[plane + idx] = p.G / 255f;
                        input[2 * plane + idx] = p.B / 255f;
                    }
                }
            });
        }

        // --- Inference ---
        var tensor = new DenseTensor<float>(input, new[] { 1, 3, ImgSz, ImgSz });
        var inputs = new[] { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };
        using var results = _session.Run(inputs);

        Tensor<float> output = results.First().AsTensor<float>();
        // Expecting [1, 5, 8400]: rows are (cx, cy, w, h, score) across anchors.
        int channels = output.Dimensions[1]; // 5
        int anchors = output.Dimensions[2];   // 8400
        float[] flat = output.ToArray();      // row-major: c * anchors + a

        // --- Decode: filter -> xyxy -> undo letterbox -> clamp -> drop degenerate ---
        var boxes = new List<DetectorBox>();
        for (int a = 0; a < anchors; a++)
        {
            float score = flat[4 * anchors + a];
            if (score < confThreshold)
                continue;

            double cx = flat[a];
            double cy = flat[anchors + a];
            double w = flat[2 * anchors + a];
            double h = flat[3 * anchors + a];

            // Undo letterbox: subtract padding, divide by scale.
            double x1 = (cx - w / 2 - padX) / scale;
            double y1 = (cy - h / 2 - padY) / scale;
            double x2 = (cx + w / 2 - padX) / scale;
            double y2 = (cy + h / 2 - padY) / scale;

            // Clamp to original image bounds.
            x1 = Math.Clamp(x1, 0, origW);
            y1 = Math.Clamp(y1, 0, origH);
            x2 = Math.Clamp(x2, 0, origW);
            y2 = Math.Clamp(y2, 0, origH);

            // Drop degenerate boxes left with non-positive size after clamping.
            if (x2 - x1 <= 0 || y2 - y1 <= 0)
                continue;

            boxes.Add(new DetectorBox(
                (float)x1, (float)y1, (float)(x2 - x1), (float)(y2 - y1), score));
        }

        return NonMaxSuppression(boxes, iouThreshold);
    }

    /// <summary>
    /// Greedy NMS matching check_yolo_onnx.py: sort by score desc, keep the top
    /// box, suppress remaining boxes whose IoU with it is >= the threshold.
    /// </summary>
    private static List<DetectorBox> NonMaxSuppression(List<DetectorBox> boxes, float iouThreshold)
    {
        var order = Enumerable.Range(0, boxes.Count)
            .OrderByDescending(i => boxes[i].Score)
            .ToList();

        var keep = new List<DetectorBox>();
        while (order.Count > 0)
        {
            int best = order[0];
            DetectorBox a = boxes[best];
            keep.Add(a);
            order.RemoveAt(0);
            order.RemoveAll(i => Iou(a, boxes[i]) >= iouThreshold);
        }
        return keep;
    }

    private static float Iou(DetectorBox a, DetectorBox b)
    {
        float ax2 = a.X + a.Width, ay2 = a.Y + a.Height;
        float bx2 = b.X + b.Width, by2 = b.Y + b.Height;

        float interX1 = Math.Max(a.X, b.X);
        float interY1 = Math.Max(a.Y, b.Y);
        float interX2 = Math.Min(ax2, bx2);
        float interY2 = Math.Min(ay2, by2);

        float interW = Math.Max(interX2 - interX1, 0f);
        float interH = Math.Max(interY2 - interY1, 0f);
        float inter = interW * interH;

        float areaA = a.Width * a.Height;
        float areaB = b.Width * b.Height;
        return inter / (areaA + areaB - inter + 1e-9f);
    }

    public void Dispose() => _session.Dispose();
}
