namespace TsrApp.Models;

/// <summary>
/// One detected-and-classified traffic sign: the detector box plus the
/// classifier's verdict. DetectorScore is the box confidence from the detector;
/// Confidence is the classifier's confidence for the chosen class.
/// </summary>
public readonly record struct DetectedSign(
    DetectorBox Box,
    int ClassId,
    string ClassName,
    float Confidence,
    float DetectorScore);
