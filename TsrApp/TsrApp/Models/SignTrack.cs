namespace TsrApp.Models;

/// <summary>
/// Immutable snapshot of one active sign track for the UI overlay: the last
/// confirmed box plus the smoothed (majority-voted) class and confidence, the
/// track id, bookkeeping frame numbers, and the motion data
/// (<see cref="Velocity"/> + <see cref="LastConfirmedTimeMs"/>) the UI uses to
/// extrapolate the box between inference results.
/// </summary>
public sealed record SignTrack(
    int Id,
    DetectorBox Box,
    int ClassId,
    string ClassName,
    float Confidence,
    int FirstFrame,
    int LastConfirmedFrame,
    int Confirmations,
    BoxVelocity Velocity,
    long LastConfirmedTimeMs);
