namespace TsrApp.Models;

/// <summary>
/// Final summary of a track at the moment it closes: the smoothed class and
/// confidence, the last matched box, plus its lifespan (first/last confirmed
/// frame, confirmation count). Produced by <see cref="TsrApp.Services.SignTracker"/>
/// for CSV logging.
/// </summary>
public sealed record TrackSummary(
    int Id,
    DetectorBox Box,
    int ClassId,
    string ClassName,
    float Confidence,
    int FirstFrame,
    int LastConfirmedFrame,
    int Confirmations);
