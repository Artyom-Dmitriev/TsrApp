using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace TsrApp.Converters;

/// <summary>
/// Maps a classifier confidence (0..1) to a brush using the same thresholds as
/// the classifier-mode result badge: >0.9 green, >0.7 yellow, else red.
/// </summary>
public sealed class ConfidenceToColorConverter : IValueConverter
{
    private static readonly SolidColorBrush Green = new((Color)ColorConverter.ConvertFromString("#2E7D32")!);
    private static readonly SolidColorBrush Yellow = new((Color)ColorConverter.ConvertFromString("#F9A825")!);
    private static readonly SolidColorBrush Red = new((Color)ColorConverter.ConvertFromString("#C62828")!);

    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        float c = value switch
        {
            float f => f,
            double d => (float)d,
            _ => 0f,
        };
        return c > 0.9f ? Green : c > 0.7f ? Yellow : Red;
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
