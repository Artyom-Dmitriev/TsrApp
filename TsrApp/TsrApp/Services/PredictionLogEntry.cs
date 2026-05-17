namespace TsrApp.Services;

public sealed class PredictionLogEntry
{
    public string Timestamp { get; set; } = "";
    public string ImagePath { get; set; } = "";
    public int PredictedClassId { get; set; }
    public string PredictedClassName { get; set; } = "";
    public float Confidence { get; set; }
}
