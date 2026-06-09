namespace TsrApp.Models;

/// <summary>
/// A detected traffic-sign box in pixel coordinates of the ORIGINAL image.
/// X, Y is the top-left corner; Width, Height the box size; Score the detector
/// confidence.
/// </summary>
public readonly record struct DetectorBox(float X, float Y, float Width, float Height, float Score);
