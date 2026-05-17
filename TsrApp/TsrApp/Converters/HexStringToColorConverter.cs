using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace TsrApp.Converters;

public sealed class HexStringToColorConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is string s && !string.IsNullOrWhiteSpace(s))
        {
            try
            {
                var color = (Color)ColorConverter.ConvertFromString(s)!;
                return new SolidColorBrush(color);
            }
            catch
            {
            }
        }
        return new SolidColorBrush(Colors.Gray);
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
