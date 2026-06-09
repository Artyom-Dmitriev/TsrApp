namespace TsrApp.Services;

public sealed class PredictionLogEntry
{
    public string Timestamp { get; set; } = "";
    public string ImagePath { get; set; } = "";
    public int PredictedClassId { get; set; }
    public string PredictedClassName { get; set; } = "";
    public float Confidence { get; set; }

    // Added in milestone 3 (detector logging). Old records migrate with these empty.
    public string Mode { get; set; } = "";       // "Classifier" / "Detector"
    public string SourceType { get; set; } = ""; // "Image" (room for "Frame" later)
    public int? FrameIndex { get; set; }          // null for still images
    public string BBox { get; set; } = "";        // "x,y,w,h"; empty for classifier
}
