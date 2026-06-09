using System.IO;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace TsrApp.Services;

public record PredictionResult(
    int ClassId,
    string ClassName,
    float Confidence,
    IReadOnlyList<(int ClassId, string ClassName, float Probability)> Top3);

public sealed class ClassifierService : IDisposable
{
    private const int Size = 224;
    private const int NumClasses = 43;

    private readonly InferenceSession _session;
    private readonly Dictionary<int, string> _labels;
    private readonly ImagePreprocessor _preprocessor;

    public ClassifierService(string modelPath, string labelsPath)
    {
        _session = new InferenceSession(modelPath);

        string json = File.ReadAllText(labelsPath);
        var raw = JsonSerializer.Deserialize<Dictionary<string, string>>(json)
                  ?? throw new InvalidDataException($"Cannot parse labels at {labelsPath}");
        _labels = raw.ToDictionary(kv => int.Parse(kv.Key), kv => kv.Value);

        _preprocessor = new ImagePreprocessor();
    }

    public PredictionResult Predict(string imagePath)
    {
        float[] input = _preprocessor.LoadAndPreprocess(imagePath);
        return PredictFromInput(input);
    }

    public PredictionResult Predict(Image<Rgb24> crop)
    {
        // Does not dispose the caller's crop; Preprocess works on a clone.
        float[] input = _preprocessor.Preprocess(crop);
        return PredictFromInput(input);
    }

    private PredictionResult PredictFromInput(float[] input)
    {
        float[] logits = RunModel(input);
        float[] probs = Softmax(logits);

        int top1 = 0;
        for (int i = 1; i < probs.Length; i++)
            if (probs[i] > probs[top1]) top1 = i;

        var top3 = probs
            .Select((p, i) => (ClassId: i, ClassName: _labels[i], Probability: p))
            .OrderByDescending(t => t.Probability)
            .Take(3)
            .ToList();

        return new PredictionResult(top1, _labels[top1], probs[top1], top3);
    }

    internal float[] RunModel(float[] input)
    {
        var tensor = new DenseTensor<float>(input, new[] { 1, 3, Size, Size });
        var inputs = new[] { NamedOnnxValue.CreateFromTensor("input", tensor) };
        using var results = _session.Run(inputs);
        return results.First().AsTensor<float>().ToArray();
    }

    internal static float[] Softmax(float[] logits)
    {
        float max = logits.Max();
        float[] exps = new float[logits.Length];
        float sum = 0f;
        for (int i = 0; i < logits.Length; i++)
        {
            exps[i] = MathF.Exp(logits[i] - max);
            sum += exps[i];
        }
        for (int i = 0; i < exps.Length; i++)
            exps[i] /= sum;
        return exps;
    }

    public void Dispose() => _session.Dispose();
}
