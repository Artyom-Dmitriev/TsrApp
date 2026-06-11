namespace TsrApp.Models;

/// <summary>
/// Per-millisecond rate of change of a track's box, estimated from its two last
/// confirmed positions. Width/height move too — signs grow as they approach.
/// Used to extrapolate the box between inference results.
/// </summary>
public readonly record struct BoxVelocity(float Vx, float Vy, float Vw, float Vh)
{
    public static readonly BoxVelocity Zero = new(0f, 0f, 0f, 0f);
}
