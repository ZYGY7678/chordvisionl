"""Chromagram extraction: from a CQT to a 12-dimensional pitch class profile.

A chromagram collapses every octave onto one of twelve pitch classes
(C, C#, D ... B).  Chords are defined by pitch classes, not by absolute
pitches, so this is the representation chord recognition works in.

Two things are done here that are often skipped and that matter a lot on
real recordings:

1. **Tuning estimation.**  Plenty of recordings are not at A=440 - tape
   speed, analogue mastering, a guitar tuned by ear, or a deliberate
   A=432/A=415 tuning.  If we fold the CQT onto pitch classes assuming
   A=440 when the record is 40 cents flat, every note smears across two
   pitch classes and the chord estimates fall apart.  We measure the
   offset from the spectrum itself and compensate.

2. **Log compression.**  Chord tones other than the root are frequently
   20-30 dB quieter than the loudest element of a mix.  Working on raw
   magnitudes lets the loudest voice dominate the profile; compressing
   with log(1 + gamma*x) brings the quieter chord tones into range
   without amplifying the noise floor the way a plain dB conversion does.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import median_filter

from .cqt import CQTSpec

PITCH_CLASSES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCH_CLASSES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def estimate_tuning(
    C: np.ndarray,
    spec: CQTSpec,
    min_peak_ratio: float = 0.05,
) -> float:
    """Estimate how far the recording sits from equal temperament at A=440.

    Returns the offset in **cents**, in the range (-50, +50].  A return of
    -40 means the record sounds 40 cents flat and every pitch class should
    be read 40 cents lower than the A=440 grid.

    Method: find the spectral peaks of every frame, refine each peak to
    sub-bin precision by fitting a parabola to its neighbours, express the
    refined position relative to the nearest semitone of the A=440 grid,
    and take the energy-weighted circular mean of those deviations.  A
    circular mean is required because the deviation wraps: +49 cents and
    -49 cents are nearly the same place, and an arithmetic mean of the two
    would wrongly return 0.
    """
    if C.size == 0:
        return 0.0

    bins_per_semitone = spec.bins_per_octave / 12.0
    if bins_per_semitone < 2:
        # Not enough resolution to measure detuning at all.
        return 0.0

    # Peak picking along the frequency axis, per frame.
    prev = C[:-2, :]
    curr = C[1:-1, :]
    nxt = C[2:, :]
    is_peak = (curr > prev) & (curr >= nxt)

    frame_max = C.max(axis=0, keepdims=True)
    is_peak &= curr > (min_peak_ratio * frame_max)

    bin_idx, frame_idx = np.nonzero(is_peak)
    if bin_idx.size == 0:
        return 0.0
    bin_idx = bin_idx + 1  # undo the slice offset

    a = C[bin_idx - 1, frame_idx]
    b = C[bin_idx, frame_idx]
    c = C[bin_idx + 1, frame_idx]
    denom = (a - 2.0 * b + c)
    shift = np.where(np.abs(denom) > 1e-12, 0.5 * (a - c) / np.where(denom == 0, 1e-12, denom), 0.0)
    shift = np.clip(shift, -0.5, 0.5)

    refined = bin_idx.astype(np.float64) + shift

    # Position within a semitone, as a fraction of a semitone in [0, 1).
    frac = np.mod(refined / bins_per_semitone, 1.0)
    # Centre on zero: [-0.5, 0.5)
    dev = np.where(frac > 0.5, frac - 1.0, frac)

    weights = b.astype(np.float64)
    if weights.sum() <= 0:
        return 0.0

    # Circular mean over one semitone of phase.
    angles = 2.0 * np.pi * dev
    mean_angle = np.arctan2(
        np.sum(weights * np.sin(angles)),
        np.sum(weights * np.cos(angles)),
    )
    offset_semitones = mean_angle / (2.0 * np.pi)
    return float(offset_semitones * 100.0)


def _fold_matrix(
    spec: CQTSpec,
    tuning_cents: float,
    octave_weights: Optional[np.ndarray] = None,
    bin_range: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Build the (12, n_bins) matrix that folds CQT bins onto pitch classes.

    The tuning offset is applied as a fractional bin shift, spread across
    the two neighbouring bins by linear interpolation, so a recording that
    is 40 cents flat is read correctly instead of being quantised to the
    nearest bin.
    """
    bps = spec.bins_per_octave / 12.0          # bins per semitone
    shift = (tuning_cents / 100.0) * bps        # in bins

    lo, hi = bin_range if bin_range is not None else (0, spec.n_bins)
    lo = max(0, lo)
    hi = min(spec.n_bins, hi)

    M = np.zeros((12, spec.n_bins), dtype=np.float64)

    n_oct = spec.n_bins // spec.bins_per_octave
    for octave in range(n_oct):
        w_oct = 1.0 if octave_weights is None else float(octave_weights[octave])
        if w_oct <= 0:
            continue
        for pc in range(12):
            centre = octave * spec.bins_per_octave + pc * bps + shift
            b0 = int(np.floor(centre))
            frac = centre - b0
            for b, w in ((b0, 1.0 - frac), (b0 + 1, frac)):
                if w <= 0:
                    continue
                if lo <= b < hi:
                    M[pc, b] += w * w_oct
    return M


def _octave_weights(
    spec: CQTSpec,
    centre_hz: float = 329.63,   # E4
    sigma_octaves: float = 1.25,
) -> np.ndarray:
    """Gaussian emphasis on the register where chords actually live.

    Very low bins are dominated by the bass line and the kick drum; very
    high bins by cymbals, sibilance and the upper partials of everything.
    The harmonic content that identifies a chord sits in the middle, so we
    weight the octaves with a wide Gaussian rather than treating them all
    equally.  The weighting is deliberately wide: it de-emphasises the
    extremes, it does not discard them.
    """
    n_oct = spec.n_bins // spec.bins_per_octave
    centres = spec.f_min * np.power(2.0, np.arange(n_oct) + 0.5)
    d = np.log2(centres / centre_hz)
    return np.exp(-0.5 * (d / sigma_octaves) ** 2)


def chromagram(
    C: np.ndarray,
    spec: CQTSpec,
    tuning_cents: float = 0.0,
    gamma: float = 100.0,
    use_octave_weighting: bool = True,
    bin_range: Optional[Tuple[int, int]] = None,
    smooth_frames: int = 0,
    normalize: Optional[str] = "l2",
    octave_centre_hz: float = 329.63,
    octave_sigma: float = 1.25,
) -> np.ndarray:
    """Fold a CQT magnitude array into a (12, n_frames) chromagram.

    Parameters
    ----------
    gamma : strength of the log compression, ``log(1 + gamma * x)``.  Larger
        values lift quiet chord tones further.  0 disables compression.
    smooth_frames : if > 1, apply a median filter of this length along time.
        A median filter is used rather than a mean because it removes short
        spikes (a cymbal crash, a passing note) without blurring the edges
        of genuine chord changes.
    normalize : "l2", "l1", "max" or None, applied per frame.
    """
    if C.shape[0] != spec.n_bins:
        raise ValueError("CQT array does not match the supplied CQTSpec")

    X = C
    if gamma and gamma > 0:
        X = np.log1p(gamma * X)

    weights = (
        _octave_weights(spec, octave_centre_hz, octave_sigma)
        if use_octave_weighting else None
    )
    M = _fold_matrix(spec, tuning_cents, weights, bin_range)
    chroma = M @ X

    if smooth_frames and smooth_frames > 1:
        chroma = median_filter(chroma, size=(1, int(smooth_frames)), mode="nearest")

    if normalize:
        chroma = normalize_frames(chroma, normalize)
    return chroma


def normalize_frames(X: np.ndarray, kind: str = "l2", eps: float = 1e-9) -> np.ndarray:
    """Normalise each column (frame) of ``X``.

    Frames that are essentially silent are left as zeros rather than being
    amplified into noise - that is what lets the decoder recognise silence
    as "no chord" instead of hallucinating one.
    """
    if kind == "l2":
        n = np.sqrt(np.sum(X * X, axis=0, keepdims=True))
    elif kind == "l1":
        n = np.sum(np.abs(X), axis=0, keepdims=True)
    elif kind == "max":
        n = np.max(np.abs(X), axis=0, keepdims=True)
    else:
        raise ValueError(f"unknown normalisation: {kind}")

    out = np.zeros_like(X)
    ok = (n > eps).ravel()
    out[:, ok] = X[:, ok] / n[:, ok]
    return out


def frame_energy(C: np.ndarray, spec: CQTSpec, bin_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Per-frame energy, used to tell real music from silence."""
    lo, hi = bin_range if bin_range is not None else (0, spec.n_bins)
    return np.sqrt(np.sum(C[lo:hi, :] ** 2, axis=0))


def bass_chromagram(
    C: np.ndarray,
    spec: CQTSpec,
    tuning_cents: float = 0.0,
    low_hz: float = 32.7,
    high_hz: float = 261.63,
    gamma: float = 100.0,
    smooth_frames: int = 0,
) -> np.ndarray:
    """A chromagram restricted to the bass register (C1 to C4 by default).

    The bass note is what decides whether a chord is in root position or an
    inversion, and it lives in a register where the rest of the arrangement
    barely reaches.  Computing it separately, with no octave weighting, is
    far more reliable than trying to read the bass out of the full-range
    chromagram.
    """
    freqs = spec.bin_frequencies()
    lo = int(np.searchsorted(freqs, low_hz, side="left"))
    hi = int(np.searchsorted(freqs, high_hz, side="right"))
    return chromagram(
        C, spec,
        tuning_cents=tuning_cents,
        gamma=gamma,
        use_octave_weighting=False,
        bin_range=(lo, hi),
        smooth_frames=smooth_frames,
        normalize="l2",
    )
