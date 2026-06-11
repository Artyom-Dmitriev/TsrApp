using TsrApp.Models;

namespace TsrApp.Services;

/// <summary>Result of one <see cref="SignTracker.Update"/> call.</summary>
public sealed record TrackerResult(
    IReadOnlyList<SignTrack> Active,
    IReadOnlyList<TrackSummary> Closed);

/// <summary>
/// Greedy IoU tracker over inference results. Matches incoming detections to
/// active tracks by box IoU, opens tracks for unmatched detections, and keeps
/// unmatched tracks alive for a few results before closing them — which removes
/// flicker from single-frame detector misses. Each track smooths its class by a
/// majority vote over its recent classifications.
///
/// Pure logic: no UI, no threads, no shared mutable state beyond its own track
/// list. Intended to be driven from a single thread (the inference thread).
/// </summary>
public sealed class SignTracker
{
    // --- Tunables (hand-tuned; keep them together) ---
    // Min box IoU for a detection to be considered the same physical sign. Low
    // (0.18) because boxes drift noticeably between inference results at the
    // current rate; a higher threshold tears tracks apart and spawns duplicates.
    private const float IouMatchThreshold = 0.18f;
    // How many consecutive un-confirmed results a track survives before closing.
    private const int MaxMisses = 2;
    // Number of recent classifications kept per track for the majority vote.
    private const int ClassHistorySize = 5;
    // A track is shown (and logged on close) only after this many confirmations;
    // weeds out single-frame false positives.
    private const int MinConfirmationsToDisplay = 2;
    // A track not confirmed within this many ms is hidden from the overlay (but
    // stays alive until MaxMisses closes it, and is still logged as before).
    private const long StaleTrackHideMs = 700;
    // Min smoothed confidence for a track to be shown and logged. Real signs read
    // ~90–100% as they approach; detector hallucinations top out around ~68%, and
    // the occasional mis-classification also sits below this. A high bar (0.75)
    // sends both to the bin cleanly. Tracks below it are hidden/unlogged but stay
    // alive, so they can "rehabilitate" (start showing + logging) if confidence
    // rises later as the sign approaches.
    private const float MinTrackConfidence = 0.75f;
    // Extrapolation cap: past this many ms since the last confirmation, the box
    // freezes at its last predicted position (and is soon hidden by StaleTrackHideMs).
    private const long MaxExtrapolationMs = 600;
    // If less than this elapsed between two confirmations, treat the velocity as
    // unreliable (and avoid dividing by a tiny dt) -> zero it.
    private const long MinVelocityDtMs = 5;
    // If the box center jumps more than this multiple of its size between two
    // confirmations, it's likely a re-match to a different sign -> zero velocity.
    private const float MaxJumpFactor = 2.0f;
    // EMA factor for velocity: V = a*measured + (1-a)*previous. Lower = smoother
    // (more inertia, less jitter); higher = more responsive to real acceleration.
    // Biased toward responsiveness (0.55): the box lagged on the perspective
    // acceleration signs pick up near the frame edges.
    private const float VelocitySmoothing = 0.55f;
    // Position blend at confirmation: box = b*detection + (1-b)*predicted. Lower =
    // smoother/laggier; higher = snappier to the raw detection. Biased toward
    // responsiveness (0.75) to cut the same edge-of-frame lag.
    private const float PositionSmoothing = 0.75f;
    // Scales the width/height velocity during extrapolation only. Position moves at
    // full speed; size barely drifts between confirmations (small-box size noise is
    // large and its extrapolation makes boxes "breathe"). 0 freezes size entirely.
    private const float SizeExtrapolationDamping = 0.3f;

    private readonly List<Track> _tracks = new();
    private int _nextId;

    /// <summary>
    /// Feeds one inference result (detections for <paramref name="frameIndex"/>,
    /// observed at <paramref name="nowMs"/> monotonic milliseconds), updates the
    /// tracks, and returns the current active snapshot plus any tracks that closed
    /// during this call.
    /// </summary>
    public TrackerResult Update(IReadOnlyList<DetectedSign> detections, int frameIndex, long nowMs)
    {
        bool[] trackMatched = new bool[_tracks.Count];
        bool[] detMatched = new bool[detections.Count];

        // Predict each track's position at nowMs and match against that, not the
        // last confirmed box — so a moving sign still matches its detection. The
        // MaxExtrapolationMs cap inside PredictBox keeps stale tracks cautious.
        var predicted = new DetectorBox[_tracks.Count];
        for (int t = 0; t < _tracks.Count; t++)
        {
            Track tr = _tracks[t];
            predicted[t] = PredictBox(tr.Box, tr.Velocity, tr.LastConfirmedTimeMs, nowMs);
        }

        // Build candidate pairs above the IoU threshold, best first.
        var pairs = new List<(int Track, int Det, float Iou)>();
        for (int t = 0; t < _tracks.Count; t++)
        {
            for (int d = 0; d < detections.Count; d++)
            {
                float iou = Iou(predicted[t], detections[d].Box);
                if (iou >= IouMatchThreshold)
                    pairs.Add((t, d, iou));
            }
        }
        pairs.Sort((a, b) => b.Iou.CompareTo(a.Iou));

        // Greedy: take the highest-IoU pair whose track and detection are both free.
        foreach (var p in pairs)
        {
            if (trackMatched[p.Track] || detMatched[p.Det])
                continue;
            trackMatched[p.Track] = true;
            detMatched[p.Det] = true;
            _tracks[p.Track].Confirm(detections[p.Det], frameIndex, nowMs);
        }

        // Unmatched detections open new tracks.
        for (int d = 0; d < detections.Count; d++)
        {
            if (detMatched[d])
                continue;
            var track = new Track(++_nextId, frameIndex);
            track.Confirm(detections[d], frameIndex, nowMs);
            _tracks.Add(track);
        }

        // Unmatched tracks miss; close the ones that have missed too long. Only the
        // original tracks (indices < trackMatched.Length) are considered — the new
        // tracks appended above were just confirmed. Iterate downward so RemoveAt
        // doesn't disturb indices we haven't visited yet.
        var closed = new List<TrackSummary>();
        for (int t = trackMatched.Length - 1; t >= 0; t--)
        {
            if (trackMatched[t])
                continue;

            Track track = _tracks[t];
            track.Misses++;
            if (track.Misses > MaxMisses)
            {
                // Filtered tracks (too few confirmations or low confidence) are
                // dropped silently — not logged.
                TrackSummary? summary = TryBuildSummary(track);
                if (summary is not null)
                    closed.Add(summary);
                _tracks.RemoveAt(t);
            }
        }

        var active = new List<SignTrack>();
        foreach (Track track in _tracks)
        {
            SignTrack? snapshot = TryBuildSnapshot(track, nowMs);
            if (snapshot is not null)
                active.Add(snapshot);
        }

        return new TrackerResult(active, closed);
    }

    /// <summary>
    /// Closes all active tracks and clears state. Returns summaries of the tracks
    /// that pass the display/confidence filters (others are dropped as noise).
    /// </summary>
    public IReadOnlyList<TrackSummary> Reset()
    {
        var closed = new List<TrackSummary>();
        foreach (Track track in _tracks)
        {
            TrackSummary? summary = TryBuildSummary(track);
            if (summary is not null)
                closed.Add(summary);
        }
        _tracks.Clear();
        _nextId = 0;
        return closed;
    }

    /// <summary>
    /// Predicts a track's box at <paramref name="nowMs"/> by extrapolating along
    /// its velocity, capped at <see cref="MaxExtrapolationMs"/> (the box freezes
    /// beyond that). Pure function for the UI to call each rendered frame.
    /// </summary>
    public static DetectorBox PredictBox(SignTrack t, long nowMs)
        => PredictBox(t.Box, t.Velocity, t.LastConfirmedTimeMs, nowMs);

    private static DetectorBox PredictBox(DetectorBox box, BoxVelocity v, long lastConfirmedMs, long nowMs)
    {
        long dt = nowMs - lastConfirmedMs;
        if (dt < 0) dt = 0;
        if (dt > MaxExtrapolationMs) dt = MaxExtrapolationMs;

        float x = box.X + v.Vx * dt;
        float y = box.Y + v.Vy * dt;
        // Size barely drifts between confirmations (damped); position is full speed.
        float w = Math.Max(1f, box.Width + v.Vw * dt * SizeExtrapolationDamping);
        float h = Math.Max(1f, box.Height + v.Vh * dt * SizeExtrapolationDamping);
        return new DetectorBox(x, y, w, h, box.Score);
    }

    /// <summary>
    /// Build an overlay snapshot, or null if the track is filtered out: too few
    /// confirmations, smoothed confidence below the threshold, or stale (not
    /// confirmed within <see cref="StaleTrackHideMs"/>).
    /// </summary>
    private static SignTrack? TryBuildSnapshot(Track t, long nowMs)
    {
        if (t.Confirmations < MinConfirmationsToDisplay)
            return null;
        (int classId, string className, float confidence) = t.SmoothedClass();
        if (confidence < MinTrackConfidence)
            return null;
        if (nowMs - t.LastConfirmedTimeMs > StaleTrackHideMs)
            return null; // stale: hide but keep alive

        return new SignTrack(
            t.Id, t.Box, classId, className, confidence,
            t.FirstFrame, t.LastConfirmedFrame, t.Confirmations,
            t.Velocity, t.LastConfirmedTimeMs);
    }

    /// <summary>
    /// Build a closing summary, or null if the track is filtered out: too few
    /// confirmations or smoothed confidence below the threshold.
    /// </summary>
    private static TrackSummary? TryBuildSummary(Track t)
    {
        if (t.Confirmations < MinConfirmationsToDisplay)
            return null;
        (int classId, string className, float confidence) = t.SmoothedClass();
        if (confidence < MinTrackConfidence)
            return null;

        return new TrackSummary(
            t.Id, t.Box, classId, className, confidence,
            t.FirstFrame, t.LastConfirmedFrame, t.Confirmations);
    }

    private static float Iou(DetectorBox a, DetectorBox b)
    {
        float ax2 = a.X + a.Width, ay2 = a.Y + a.Height;
        float bx2 = b.X + b.Width, by2 = b.Y + b.Height;

        float interX1 = Math.Max(a.X, b.X);
        float interY1 = Math.Max(a.Y, b.Y);
        float interX2 = Math.Min(ax2, bx2);
        float interY2 = Math.Min(ay2, by2);

        float interW = Math.Max(interX2 - interX1, 0f);
        float interH = Math.Max(interY2 - interY1, 0f);
        float inter = interW * interH;

        float areaA = a.Width * a.Height;
        float areaB = b.Width * b.Height;
        return inter / (areaA + areaB - inter + 1e-9f);
    }

    private sealed class Track
    {
        private readonly List<(int ClassId, string ClassName, float Confidence)> _history = new();

        public Track(int id, int firstFrame)
        {
            Id = id;
            FirstFrame = firstFrame;
        }

        public int Id { get; }
        public int FirstFrame { get; }
        public DetectorBox Box { get; private set; }
        public int LastConfirmedFrame { get; private set; }
        public long LastConfirmedTimeMs { get; private set; }
        public int Confirmations { get; private set; }
        public int Misses { get; set; }
        public BoxVelocity Velocity { get; private set; } = BoxVelocity.Zero;

        /// <summary>Apply a matched detection: move the box, record the class, reset misses.</summary>
        public void Confirm(DetectedSign det, int frameIndex, long nowMs)
        {
            // Smooth the box toward the detection, blending with where the track
            // predicted itself to be, and derive velocity from the smoothed path.
            // First confirmation or a re-match jump take the detection outright,
            // with no position blend and zero velocity.
            DetectorBox newBox;
            if (Confirmations == 0 || IsImplausibleJump(Box, det.Box))
            {
                newBox = det.Box;
                Velocity = BoxVelocity.Zero;
            }
            else
            {
                DetectorBox predicted = PredictBox(Box, Velocity, LastConfirmedTimeMs, nowMs);
                newBox = Blend(det.Box, predicted, PositionSmoothing); // b*detection + (1-b)*predicted

                BoxVelocity? measured = EstimateVelocity(Box, newBox, nowMs - LastConfirmedTimeMs);
                Velocity = measured is null
                    ? BoxVelocity.Zero
                    : Blend(measured.Value, Velocity, VelocitySmoothing);
            }

            Box = newBox;
            _history.Add((det.ClassId, det.ClassName, det.Confidence));
            if (_history.Count > ClassHistorySize)
                _history.RemoveAt(0);
            LastConfirmedFrame = frameIndex;
            LastConfirmedTimeMs = nowMs;
            Confirmations++;
            Misses = 0;
        }

        /// <summary>
        /// Per-ms velocity from <paramref name="prev"/> to <paramref name="curr"/>,
        /// or null if a guard trips: tiny dt (division blow-up) or an implausible
        /// jump (a re-match onto a different sign). A null tells the caller to zero
        /// the velocity rather than blend it.
        /// </summary>
        private static BoxVelocity? EstimateVelocity(DetectorBox prev, DetectorBox curr, long dtMs)
        {
            if (dtMs < MinVelocityDtMs)
                return null;
            if (IsImplausibleJump(prev, curr))
                return null;

            float inv = 1f / dtMs;
            return new BoxVelocity(
                (curr.X - prev.X) * inv,
                (curr.Y - prev.Y) * inv,
                (curr.Width - prev.Width) * inv,
                (curr.Height - prev.Height) * inv);
        }

        /// <summary>True if the box center jumped more than <see cref="MaxJumpFactor"/>
        /// times its size — likely a re-match onto a different sign.</summary>
        private static bool IsImplausibleJump(DetectorBox prev, DetectorBox curr)
        {
            float prevCx = prev.X + prev.Width / 2f;
            float prevCy = prev.Y + prev.Height / 2f;
            float currCx = curr.X + curr.Width / 2f;
            float currCy = curr.Y + curr.Height / 2f;
            float jump = MathF.Sqrt((currCx - prevCx) * (currCx - prevCx)
                                  + (currCy - prevCy) * (currCy - prevCy));
            float size = MathF.Max(prev.Width, prev.Height);
            return jump > MaxJumpFactor * size;
        }

        /// <summary>EMA blend: <c>a*measured + (1-a)*previous</c>, per component.</summary>
        private static BoxVelocity Blend(BoxVelocity measured, BoxVelocity previous, float a)
            => new(
                a * measured.Vx + (1f - a) * previous.Vx,
                a * measured.Vy + (1f - a) * previous.Vy,
                a * measured.Vw + (1f - a) * previous.Vw,
                a * measured.Vh + (1f - a) * previous.Vh);

        /// <summary>Box blend: <c>t*a + (1-t)*b</c> per component; keeps a's score.</summary>
        private static DetectorBox Blend(DetectorBox a, DetectorBox b, float t)
            => new(
                t * a.X + (1f - t) * b.X,
                t * a.Y + (1f - t) * b.Y,
                t * a.Width + (1f - t) * b.Width,
                t * a.Height + (1f - t) * b.Height,
                a.Score);

        /// <summary>
        /// Majority vote over recent classifications. Ties break toward the class
        /// with the greater summed confidence; the reported confidence is the
        /// average over the winning class's votes.
        /// </summary>
        public (int ClassId, string ClassName, float Confidence) SmoothedClass()
        {
            var winner = _history
                .GroupBy(h => h.ClassId)
                .Select(g => (
                    ClassId: g.Key,
                    ClassName: g.First().ClassName,
                    Count: g.Count(),
                    SumConf: g.Sum(x => x.Confidence)))
                .OrderByDescending(g => g.Count)
                .ThenByDescending(g => g.SumConf)
                .First();

            return (winner.ClassId, winner.ClassName, winner.SumConf / winner.Count);
        }
    }
}
