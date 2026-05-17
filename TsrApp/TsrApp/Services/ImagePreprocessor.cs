using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace TsrApp.Services;

public sealed class ImagePreprocessor
{
    private const int Size = 224;
    private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] Std = { 0.229f, 0.224f, 0.225f };

    public float[] LoadAndPreprocess(string path)
    {
        using Image<Rgb24> image = Image.Load<Rgb24>(path);
        image.Mutate(ctx => ctx.Resize(Size, Size, KnownResamplers.Triangle));

        float[] data = new float[3 * Size * Size];
        int plane = Size * Size;

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < Size; y++)
            {
                Span<Rgb24> row = accessor.GetRowSpan(y);
                int rowOffset = y * Size;
                for (int x = 0; x < Size; x++)
                {
                    Rgb24 p = row[x];
                    int idx = rowOffset + x;
                    data[idx] = (p.R / 255f - Mean[0]) / Std[0];
                    data[plane + idx] = (p.G / 255f - Mean[1]) / Std[1];
                    data[2 * plane + idx] = (p.B / 255f - Mean[2]) / Std[2];
                }
            }
        });

        return data;
    }
}
