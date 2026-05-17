using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using TsrApp.Services;

namespace TsrApp.ViewModels;

public partial class MainViewModel : ObservableObject
{
    private readonly ClassifierService _classifier;
    private readonly PredictionLogger _logger;

    [ObservableProperty] private string? _imagePath;
    [ObservableProperty] private string _resultText = "";
    [ObservableProperty] private string _confidenceColor = "#999999";
    [ObservableProperty] private bool _hasResult;
    [ObservableProperty] private string _top3Text = "";

    public ObservableCollection<PredictionLogEntry> History { get; } = new();

    public MainViewModel()
    {
        string baseDir = AppDomain.CurrentDomain.BaseDirectory;
        _classifier = new ClassifierService(
            Path.Combine(baseDir, "Assets", "model.onnx"),
            Path.Combine(baseDir, "Assets", "labels.json"));
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
}
