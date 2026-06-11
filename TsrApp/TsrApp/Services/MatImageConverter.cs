using System.Runtime.InteropServices;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using OpenCvSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace TsrApp.Services;

/// <summary>
/// Helpers for moving OpenCV <see cref="Mat"/> frames into WPF imaging types.
/// These same frames are reused by the inference stage later on.
/// </summary>
public static class MatImageConverter
{
    /// <summary>
    /// Copies a 3-channel BGR8 <see cref="Mat"/> into a frozen
    /// <see cref="BitmapSource"/>. The result owns its own pixel buffer, so it is
    /// completely independent of <paramref name="bgr"/> — the caller may dispose
    /// the Mat right after this returns. The bitmap is frozen, so it can safely be
    /// handed to the UI thread.
    /// </summary>
    public static BitmapSource ToBitmapSource(Mat bgr)
    {
        ArgumentNullException.ThrowIfNull(bgr);
        if (bgr.Empty())
            throw new ArgumentException("Frame is empty.", nameof(bgr));
        if (bgr.Type() != MatType.CV_8UC3)
            throw new ArgumentException(
                $"Expected a BGR 8-bit 3-channel frame (CV_8UC3), got {bgr.Type()}.",
                nameof(bgr));

        int width = bgr.Width;
        int height = bgr.Height;
        int stride = (int)bgr.Step();

        // Copy the native pixels into a managed buffer the WPF bitmap will own.
        byte[] pixels = new byte[stride * height];
        Marshal.Copy(bgr.Data, pixels, 0, pixels.Length);

        // VideoCapture yields BGR, which maps 1:1 onto WPF's Bgr24 format.
        var bitmap = BitmapSource.Create(
            width, height, 96, 96, PixelFormats.Bgr24, null, pixels, stride);
        bitmap.Freeze();
        return bitmap;
    }

    /// <summary>
    /// Converts a 3-channel BGR8 <see cref="Mat"/> into an independent ImageSharp
    /// <see cref="Image{Rgb24}"/> for the inference pipeline. OpenCV frames are
    /// BGR, but the detector/classifier expect RGB, so the channel order is
    /// swapped here (BGR -> RGB). The result owns its own buffer — the caller may
    /// dispose the source Mat immediately and must dispose the returned image.
    /// </summary>
    public static Image<Rgb24> ToImageSharpRgb24(Mat bgr)
    {
        ArgumentNullException.ThrowIfNull(bgr);
        if (bgr.Empty())
            throw new ArgumentException("Frame is empty.", nameof(bgr));
        if (bgr.Type() != MatType.CV_8UC3)
            throw new ArgumentException(
                $"Expected a BGR 8-bit 3-channel frame (CV_8UC3), got {bgr.Type()}.",
                nameof(bgr));

        // CvtColor allocates a fresh, continuous RGB Mat; release it right after
        // copying so we don't leak one Mat per processed frame.
        using Mat rgb = new();
        Cv2.CvtColor(bgr, rgb, ColorConversionCodes.BGR2RGB);

        int width = rgb.Width;
        int height = rgb.Height;
        byte[] pixels = new byte[width * height * 3];
        Marshal.Copy(rgb.Data, pixels, 0, pixels.Length);

        return Image.LoadPixelData<Rgb24>(pixels, width, height);
    }
}
