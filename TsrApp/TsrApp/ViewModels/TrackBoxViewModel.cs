using CommunityToolkit.Mvvm.ComponentModel;
using TsrApp.Models;
using TsrApp.Services;

namespace TsrApp.ViewModels;

/// <summary>
/// One overlay box for a live track. Its position/size are mutable so they can be
/// re-extrapolated on every rendered video frame without rebuilding the bound
/// collection. Class/confidence/id change only when a new inference result
/// reconciles the track. Lives and is mutated on the UI thread only.
/// </summary>
public partial class TrackBoxViewModel : ObservableObject
{
    // Fraction of the box that must remain inside the frame to keep it visible.
    private const float MinVisibleFraction = 0.5f;

    [ObservableProperty] private double _x;
    [ObservableProperty] private double _y;
    [ObservableProperty] private double _width;
    [ObservableProperty] private double _height;
    [ObservableProperty] private bool _isVisible;
    [ObservableProperty] private string _className = "";
    [ObservableProperty] private float _confidence;

    public int Id { get; }

    /// <summary>Latest tracker snapshot; the source for extrapolation.</summary>
    public SignTrack Snapshot { get; private set; }

    public TrackBoxViewModel(SignTrack snapshot)
    {
        Id = snapshot.Id;
        Snapshot = snapshot;
        Apply(snapshot);
    }

    /// <summary>Adopt a fresh snapshot for the same track (new inference result).</summary>
    public void Update(SignTrack snapshot)
    {
        Snapshot = snapshot;
        Apply(snapshot);
    }

    private void Apply(SignTrack snapshot)
    {
        ClassName = snapshot.ClassName;
        Confidence = snapshot.Confidence;
    }

    /// <summary>
    /// Re-extrapolate the box for the current time and clamp visibility: hide if
    /// the frame size is unknown yet, or more than half the box has left the frame.
    /// </summary>
    public void UpdatePosition(long nowMs, int frameW, int frameH)
    {
        if (frameW <= 0 || frameH <= 0)
        {
            IsVisible = false;
            return;
        }

        DetectorBox b = SignTracker.PredictBox(Snapshot, nowMs);

        float boxArea = b.Width * b.Height;
        float visibleW = Math.Max(0f, Math.Min(b.X + b.Width, frameW) - Math.Max(b.X, 0f));
        float visibleH = Math.Max(0f, Math.Min(b.Y + b.Height, frameH) - Math.Max(b.Y, 0f));
        float visibleArea = visibleW * visibleH;

        if (boxArea <= 0f || visibleArea / boxArea < MinVisibleFraction)
        {
            IsVisible = false;
            return;
        }

        X = b.X;
        Y = b.Y;
        Width = b.Width;
        Height = b.Height;
        IsVisible = true;
    }
}
