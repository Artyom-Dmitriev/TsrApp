using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using TsrApp.Models;
using TsrApp.Services;

namespace TsrApp.ViewModels;

public enum AppMode
{
    Classifier = 0,
    Detector = 1,
    Video = 2,
}

public partial class MainViewModel : ObservableObject, IDisposable
{
    private readonly ClassifierService _classifier;
    private readonly DetectorService _detector;
    private readonly PipelineService _pipeline;
    private readonly PredictionLogger _logger;

    [ObservableProperty] private string? _imagePath;
    [ObservableProperty] private string _resultText = "";
    [ObservableProperty] private string _confidenceColor = "#999999";
    [ObservableProperty] private bool _hasResult;
    [ObservableProperty] private string _top3Text = "";

    // Mode switch: bound to TabControl.SelectedIndex; CurrentMode is the typed view.
    [ObservableProperty] private int _selectedModeIndex;
    public AppMode CurrentMode => (AppMode)SelectedModeIndex;
    partial void OnSelectedModeIndexChanged(int value) => OnPropertyChanged(nameof(CurrentMode));

    // Detector mode state.
    [ObservableProperty] private string? _detectorImagePath;
    [ObservableProperty] private ImageSource? _detectorImageSource;
    [ObservableProperty] private int _detectorImageWidth;
    [ObservableProperty] private int _detectorImageHeight;
    [ObservableProperty] private bool _noSignsFound;

    public ObservableCollection<DetectedSign> Detections { get; } = new();

    public ObservableCollection<PredictionLogEntry> History { get; } = new();

    public MainViewModel()
    {
        string baseDir = AppDomain.CurrentDomain.BaseDirectory;
        _classifier = new ClassifierService(
            Path.Combine(baseDir, "Assets", "model.onnx"),
            Path.Combine(baseDir, "Assets", "labels.json"));
        _detector = new DetectorService(Path.Combine(baseDir, "Assets", "detector.onnx"));
        _pipeline = new PipelineService(_detector, _classifier);
        _logger = new PredictionLogger(Path.Combine(baseDir, "predictions_log.csv"));

        foreach (var e in _logger.ReadAll().OrderByDescending(x => x.Timestamp))
            History.Add(e);
    }

    [RelayCommand]
    private void LoadImage()
    {
        var dlg = new OpenFileDialog
        {
            Title = "Выберите изображение знака",
            Filter = "Изображения (*.png;*.jpg;*.jpeg;*.bmp)|*.png;*.jpg;*.jpeg;*.bmp"
                   + "|Все файлы (*.*)|*.*",
        };
        if (dlg.ShowDialog() != true) return;

        PredictionResult result = _classifier.Predict(dlg.FileName);

        ImagePath = dlg.FileName;
        ResultText = $"{result.ClassName}  ({result.Confidence * 100f:0.0}%)";
        ConfidenceColor = result.Confidence > 0.9f ? "#2E7D32"
                        : result.Confidence > 0.7f ? "#F9A825"
                        : "#C62828";
        Top3Text = string.Join(Environment.NewLine,
            result.Top3.Select(t => $"{t.ClassName}: {t.Probability * 100f:0.0}%"));
        HasResult = true;

        var entry = new PredictionLogEntry
        {
            Timestamp = DateTime.UtcNow.ToString("O"),
            ImagePath = dlg.FileName,
            PredictedClassId = result.ClassId,
            PredictedClassName = result.ClassName,
            Confidence = result.Confidence,
            Mode = "Classifier",
            SourceType = "Image",
        };
        _logger.Append(entry);
        History.Insert(0, entry);
    }

    [RelayCommand]
    private void ClearHistory()
    {
        var answer = MessageBox.Show(
            "Очистить всю историю предсказаний? Действие нельзя отменить.",
            "Подтверждение",
            MessageBoxButton.YesNo,
            MessageBoxImage.Question);
        if (answer != MessageBoxResult.Yes) return;

        _logger.Clear();
        History.Clear();
        ImagePath = null;
        ResultText = "";
        Top3Text = "";
        ConfidenceColor = "#999999";
        HasResult = false;
    }

    [RelayCommand]
    private void DetectImage()
    {
        var dlg = new OpenFileDialog
        {
            Title = "Выберите изображение со знаками",
            Filter = "Изображения (*.png;*.jpg;*.jpeg;*.bmp)|*.png;*.jpg;*.jpeg;*.bmp"
                   + "|Все файлы (*.*)|*.*",
        };
        if (dlg.ShowDialog() != true) return;

        IReadOnlyList<DetectedSign> signs;
        ImageSource imageSource;
        using (Image<Rgb24> frame = Image.Load<Rgb24>(dlg.FileName))
        {
            DetectorImageWidth = frame.Width;
            DetectorImageHeight = frame.Height;
            signs = _pipeline.Process(frame);
            imageSource = ToBitmapSource(frame);
        }

        DetectorImageSource = imageSource;

        Detections.Clear();
        foreach (DetectedSign s in signs)
            Detections.Add(s);

        DetectorImagePath = dlg.FileName;
        NoSignsFound = Detections.Count == 0;

        string timestamp = DateTime.UtcNow.ToString("O");
        if (signs.Count == 0)
        {
            // Log the processed frame even with no detections.
            LogDetection(new PredictionLogEntry
            {
                Timestamp = timestamp,
                ImagePath = dlg.FileName,
                Mode = "Detector",
                SourceType = "Image",
            });
        }
        else
        {
            foreach (DetectedSign s in signs)
            {
                LogDetection(new PredictionLogEntry
                {
                    Timestamp = timestamp,
                    ImagePath = dlg.FileName,
                    PredictedClassId = s.ClassId,
                    PredictedClassName = s.ClassName,
                    Confidence = s.Confidence,
                    Mode = "Detector",
                    SourceType = "Image",
                    BBox = $"{s.Box.X:0},{s.Box.Y:0},{s.Box.Width:0},{s.Box.Height:0}",
                });
            }
        }
    }

    private void LogDetection(PredictionLogEntry entry)
    {
        _logger.Append(entry);
        History.Insert(0, entry);
    }

    /// <summary>
    /// Copy the exact ImageSharp pixel buffer (the one detection ran on) into a
    /// frozen 96-DPI BitmapSource so the displayed image is pixel-for-pixel the
    /// same as the detection coordinate space. ImageSharp Rgb24 and WPF Rgb24
    /// share byte layout (R, G, B), so the copy is direct.
    /// </summary>
    private static BitmapSource ToBitmapSource(Image<Rgb24> image)
    {
        int w = image.Width, h = image.Height;
        byte[] pixels = new byte[w * h * 3];
        image.CopyPixelDataTo(pixels);

        var bmp = new WriteableBitmap(w, h, 96, 96, PixelFormats.Rgb24, null);
        bmp.WritePixels(new Int32Rect(0, 0, w, h), pixels, w * 3, 0);
        bmp.Freeze();
        return bmp;
    }

    public void Dispose()
    {
        _detector.Dispose();
        _classifier.Dispose();
    }
}
