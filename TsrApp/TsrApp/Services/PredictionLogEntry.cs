namespace TsrApp.Services;

public sealed class PredictionLogEntry
{
    public string Timestamp { get; set; } = "";
    public string ImagePath { get; set; } = "";
    public int PredictedClassId { get; set; }
    public string PredictedClassName { get; set; } = "";
    public float Confidence { get; set; }

    // Added in milestone 3 (detector logging). Old records migrate with these empty.
    public string Mode { get; set; } = "";       // "Classifier" / "Detector" / "Video"
    public string SourceType { get; set; } = ""; // "Image" / "Track"
    public int? FrameIndex { get; set; }          // null for still images; FirstFrame for tracks
    public string BBox { get; set; } = "";        // "x,y,w,h"; empty for classifier

    // Added in milestone 5 (video track logging). Other-mode rows leave these empty.
    public int? TrackId { get; set; }             // track id; null for non-track rows
    public int? LastFrame { get; set; }           // last confirmed frame of the track
    public int? Confirmations { get; set; }       // number of confirmations over the track's life
}
