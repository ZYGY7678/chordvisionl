"""Onset detection, tempo estimation and beat tracking.

None of this is needed to name a chord, but all of it is needed to place a
chord *in time* convincingly:

  * chord changes in real music land on note onsets, so snapping a
    detected boundary to the nearest onset removes the small timing jitter
    the frame grid leaves behind;
  * the editor snaps user edits to the same grid;
  * the player draws the beat grid under the waveform.

The onset detector works on the CQT the chord engine already computed, so
this costs almost nothing extra.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d, maximum_filter1d


def onset_strength(C: np.ndarray, lag: int = 1, log_gamma: float = 100.0) -> np.ndarray:
    """Spectral flux onset envelope from a CQT magnitude array.

    Spectral flux measures how much energy *increased* between consecutive
    frames.  Only increases count: a note starting is an onset, a note
    stopping is not.  The log compression applied first means the measure
    responds to relative change, so a quiet passage produces onsets of
    comparable strength to a loud one.
    """
    if C.size == 0:
        return np.zeros(0, dtype=np.float64)

    S = np.log1p(log_gamma * C)
    diff = np.diff(S, n=1, axis=1, prepend=S[:, :1])
    if lag > 1:
        diff = S - np.pad(S, ((0, 0), (lag, 0)), mode="edge")[:, :S.shape[1]]

    flux = np.maximum(diff, 0.0).sum(axis=0)

    # Remove the slow drift so that peak picking works with one threshold
    # across the whole song rather than per section.
    baseline = uniform_filter1d(flux, size=max(3, int(len(flux) * 0.01) | 1), mode="nearest")
    flux = np.maximum(flux - baseline, 0.0)

    peak = float(flux.max()) if flux.size else 0.0
    if peak > 0:
        flux = flux / peak
    return flux


def pick_onsets(
    env: np.ndarray,
    frame_rate_hz: float,
    pre_max_s: float = 0.05,
    post_max_s: float = 0.03,
    pre_avg_s: float = 0.12,
    post_avg_s: float = 0.12,
    delta: float = 0.06,
    wait_s: float = 0.05,
) -> np.ndarray:
    """Pick onset frames from an onset envelope.

    A frame is an onset when it is a local maximum within a short window
    *and* exceeds the local average by ``delta``.  The local average makes
    the threshold adaptive, so a quiet verse is not skipped because a loud
    chorus set the bar.
    """
    if env.size == 0:
        return np.zeros(0, dtype=np.int64)

    def frames(sec: float) -> int:
        return max(1, int(round(sec * frame_rate_hz)))

    pre_max, post_max = frames(pre_max_s), frames(post_max_s)
    pre_avg, post_avg = frames(pre_avg_s), frames(post_avg_s)
    wait = frames(wait_s)

    local_max = maximum_filter1d(env, size=pre_max + post_max + 1, mode="nearest")
    local_avg = uniform_filter1d(env, size=pre_avg + post_avg + 1, mode="nearest")

    candidates = np.flatnonzero((env >= local_max) & (env >= local_avg + delta))

    picked: List[int] = []
    last = -np.inf
    for c in candidates:
        if c - last >= wait:
            picked.append(int(c))
            last = c
    return np.asarray(picked, dtype=np.int64)


def estimate_tempo(
    env: np.ndarray,
    frame_rate_hz: float,
    bpm_min: float = 50.0,
    bpm_max: float = 200.0,
    prior_bpm: float = 115.0,
    prior_width_octaves: float = 1.0,
) -> Optional[float]:
    """Estimate tempo by autocorrelating the onset envelope.

    If chords, drums and strums repeat every T seconds, the onset envelope
    correlates strongly with itself shifted by T.  The log-normal prior
    around 115 BPM resolves the octave ambiguity that autocorrelation
    always has - 70 BPM and 140 BPM correlate almost equally well, and
    without a prior the answer flips between them arbitrarily.
    """
    if env.size < 8:
        return None

    x = env - env.mean()
    n = len(x)
    nfft = 1
    while nfft < 2 * n:
        nfft <<= 1
    spec = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(spec * np.conj(spec), n=nfft)[:n]
    if acf[0] <= 0:
        return None
    acf = acf / acf[0]

    lag_min = max(1, int(round(frame_rate_hz * 60.0 / bpm_max)))
    lag_max = min(n - 1, int(round(frame_rate_hz * 60.0 / bpm_min)))
    if lag_max <= lag_min:
        return None

    lags = np.arange(lag_min, lag_max + 1)
    bpms = 60.0 * frame_rate_hz / lags
    prior = np.exp(-0.5 * (np.log2(bpms / prior_bpm) / prior_width_octaves) ** 2)

    score = acf[lag_min:lag_max + 1] * prior
    best = int(np.argmax(score))
    return float(bpms[best])


def track_beats(
    env: np.ndarray,
    frame_rate_hz: float,
    bpm: Optional[float] = None,
    tightness: float = 300.0,
) -> np.ndarray:
    """Dynamic-programming beat tracker (Ellis).

    Finds the sequence of beat frames that jointly maximises the onset
    strength landed on and the regularity of the spacing.  A greedy "pick
    the strongest onset every period" approach drifts as soon as one beat
    is syncopated or missing; the DP keeps the whole sequence consistent.
    """
    if env.size < 4:
        return np.zeros(0, dtype=np.int64)

    if bpm is None:
        bpm = estimate_tempo(env, frame_rate_hz)
    if not bpm or bpm <= 0:
        return np.zeros(0, dtype=np.int64)

    period = 60.0 * frame_rate_hz / bpm
    if period < 2:
        return np.zeros(0, dtype=np.int64)

    n = len(env)
    # Local score, mildly smoothed so a beat may land just beside a peak.
    localscore = uniform_filter1d(env, size=max(1, int(round(period / 8)) | 1), mode="nearest")

    backlink = np.full(n, -1, dtype=np.int64)
    cumscore = np.zeros(n, dtype=np.float64)

    window = np.arange(-int(np.round(2 * period)), -int(np.round(period / 2)) + 1)
    if window.size == 0:
        return np.zeros(0, dtype=np.int64)
    txcost = -tightness * (np.log(-window / period) ** 2)

    for t in range(n):
        idx = t + window
        valid = idx >= 0
        if not np.any(valid):
            cumscore[t] = localscore[t]
            backlink[t] = -1
            continue
        scores = cumscore[idx[valid]] + txcost[valid]
        best = int(np.argmax(scores))
        cumscore[t] = scores[best] + localscore[t]
        backlink[t] = int(idx[valid][best])

    # Start the backtrace from a strong beat near the end, not necessarily
    # the very last frame (which is often a fade-out).
    tail_start = max(0, n - int(round(2 * period)))
    tail = cumscore[tail_start:]
    if tail.size == 0:
        return np.zeros(0, dtype=np.int64)
    t = tail_start + int(np.argmax(tail))

    beats: List[int] = []
    while t >= 0:
        beats.append(int(t))
        nxt = backlink[t]
        if nxt < 0 or nxt >= t:
            break
        t = nxt
    beats.reverse()
    return np.asarray(beats, dtype=np.int64)


def snap_to_onsets(
    boundaries: np.ndarray,
    onsets: np.ndarray,
    max_shift_frames: int,
) -> np.ndarray:
    """Move each boundary to the nearest onset, if one is close enough.

    Boundaries that have no onset within ``max_shift_frames`` are left
    exactly where the decoder put them.  Snapping a boundary to a distant
    onset would be inventing precision that the audio does not support.
    """
    if boundaries.size == 0 or onsets.size == 0 or max_shift_frames <= 0:
        return boundaries

    out = boundaries.astype(np.int64).copy()
    for i, b in enumerate(out):
        j = int(np.searchsorted(onsets, b))
        best, best_d = b, max_shift_frames + 1
        for k in (j - 1, j, j + 1):
            if 0 <= k < onsets.size:
                d = abs(int(onsets[k]) - int(b))
                if d < best_d:
                    best, best_d = int(onsets[k]), d
        if best_d <= max_shift_frames:
            out[i] = best

    out = np.maximum.accumulate(out)
    return out
