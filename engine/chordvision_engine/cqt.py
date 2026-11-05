"""Constant-Q Transform, implemented here rather than taken from a library.

Why a CQT and not a plain STFT
------------------------------
A plain STFT has a fixed frequency resolution in Hz.  Musical pitch is
logarithmic: the gap between two adjacent semitones is about 3.9 Hz at C2
but about 250 Hz at C7.  A single STFT window is therefore always wrong
somewhere - either too coarse to separate low notes, or needlessly long
(and so temporally smeared) for high ones.

The Constant-Q Transform uses a separate window length per bin, chosen so
that every bin spans the same *musical* interval.  That is exactly what
chord recognition needs.

How it is computed efficiently
------------------------------
A naive CQT needs a window of ~35000 samples for the lowest bin, for every
frame.  Instead we use the standard octave-by-octave recursion:

  * build kernels **once**, for the highest octave only, where the windows
    are short;
  * analyse the top octave at the full sample rate;
  * halve the sample rate (with a proper anti-aliasing filter) and apply the
    *same* kernels again - at half the sample rate the same kernel measures
    frequencies one octave lower;
  * repeat for each octave.

Each octave therefore costs the same small amount of work, and the whole
transform is O(n log n).

The response per frame is evaluated in the frequency domain
(Brown & Puckette): for a frame ``y`` and kernel ``k`` of the same length N,

    sum_n y[n] * conj(k[n])  ==  (1/N) * sum_f Y[f] * conj(K[f])

so a single FFT per frame serves all bins of the octave at once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import resample_poly, get_window


@dataclass
class CQTSpec:
    """The parameters a CQT was computed with, kept alongside the result."""
    sample_rate: int
    f_min: float
    n_bins: int
    bins_per_octave: int
    hop_length: int
    n_octaves: int
    quality_factor: float
    frame_rate_hz: float

    def bin_frequencies(self) -> np.ndarray:
        """Centre frequency, in Hz, of every bin (index 0 is the lowest)."""
        k = np.arange(self.n_bins, dtype=np.float64)
        return self.f_min * np.power(2.0, k / self.bins_per_octave)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _build_octave_kernels(
    sr: int,
    freqs: np.ndarray,
    quality_factor: float,
    window: str = "hann",
) -> tuple[np.ndarray, int, np.ndarray]:
    """Frequency-domain kernels for one octave.

    Returns ``(K_conj, n_fft, lengths)`` where ``K_conj`` has shape
    ``(n_filters, n_fft)`` and is already conjugated, ready to be multiplied
    with the FFT of a signal frame.
    """
    lengths = np.ceil(quality_factor * sr / freqs).astype(np.int64)
    # Keep every window odd-length so it has a well defined centre sample;
    # that keeps the phase reference identical across bins.
    lengths = lengths + (1 - lengths % 2)
    n_fft = _next_pow2(int(lengths.max()))

    kernels = np.zeros((len(freqs), n_fft), dtype=np.complex128)
    for i, (f, n_len) in enumerate(zip(freqs, lengths)):
        n_len = int(n_len)
        win = get_window(window, n_len, fftbins=False).astype(np.float64)
        # Normalise so that a unit-amplitude sinusoid at f yields |X| = 0.5
        # in every bin, regardless of that bin's window length.
        win = win / np.sum(win)
        t = np.arange(n_len, dtype=np.float64) - (n_len - 1) / 2.0
        osc = np.exp(2j * np.pi * f * t / sr)
        start = (n_fft - n_len) // 2
        kernels[i, start:start + n_len] = win * osc

    K = np.fft.fft(kernels, n=n_fft, axis=1)
    return np.conjugate(K), n_fft, lengths


def _frame_fft(y: np.ndarray, n_fft: int, hop: int, n_frames: int) -> np.ndarray:
    """FFT of ``n_frames`` centred, overlapping frames of ``y``."""
    pad = n_fft // 2
    yp = np.pad(y, (pad, pad + n_fft), mode="reflect" if y.size > pad else "constant")

    idx = np.arange(n_fft)[None, :] + (np.arange(n_frames)[:, None] * hop)
    # Clamp instead of failing: the tail of the last frame may run past the
    # padded end for very short inputs.
    np.clip(idx, 0, yp.size - 1, out=idx)
    frames = yp[idx]
    return np.fft.fft(frames, n=n_fft, axis=1)


def cqt(
    y: np.ndarray,
    sr: int,
    f_min: float = 32.70319566257483,   # C1
    n_bins: int = 252,                  # 7 octaves at 36 bins/octave
    bins_per_octave: int = 36,
    hop_length: int = 512,
    filter_scale: float = 1.0,
    window: str = "hann",
) -> tuple[np.ndarray, CQTSpec]:
    """Compute the Constant-Q Transform of a mono signal.

    Parameters
    ----------
    y : 1-D float array, the mono signal.
    sr : sample rate of ``y``.
    f_min : centre frequency of the lowest bin. Default C1 (32.703 Hz).
    n_bins : total number of bins. Must be a whole number of octaves.
    bins_per_octave : 36 gives three bins per semitone, which is what the
        tuning estimator needs in order to measure detuning.
    hop_length : frame advance in samples at the input sample rate.

    Returns
    -------
    (C, spec) where ``C`` has shape ``(n_bins, n_frames)`` and holds
    **magnitudes** (not complex values), bin 0 being the lowest frequency.
    """
    if y.ndim != 1:
        raise ValueError("cqt expects a mono (1-D) signal")
    if n_bins % bins_per_octave != 0:
        raise ValueError(
            f"n_bins ({n_bins}) must be a whole number of octaves "
            f"of bins_per_octave ({bins_per_octave})"
        )
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")

    n_octaves = n_bins // bins_per_octave
    if hop_length % (2 ** (n_octaves - 1)) != 0:
        raise ValueError(
            f"hop_length ({hop_length}) must be divisible by "
            f"2**(n_octaves-1) = {2 ** (n_octaves - 1)} so that the hop stays "
            "an integer in every downsampled octave"
        )

    quality_factor = float(filter_scale) / (2.0 ** (1.0 / bins_per_octave) - 1.0)

    # Frequencies of the highest octave, which is the only one we build
    # kernels for.
    top_base = f_min * (2.0 ** (n_octaves - 1))
    k = np.arange(bins_per_octave, dtype=np.float64)
    top_freqs = top_base * np.power(2.0, k / bins_per_octave)

    nyquist = sr / 2.0
    if top_freqs[-1] >= nyquist:
        raise ValueError(
            f"The highest requested bin is {top_freqs[-1]:.1f} Hz but the Nyquist "
            f"frequency is {nyquist:.1f} Hz. Raise the sample rate or lower n_bins."
        )

    K_conj, n_fft, lengths = _build_octave_kernels(sr, top_freqs, quality_factor, window)

    n_frames = int(np.floor(y.size / hop_length)) + 1
    out = np.zeros((n_bins, n_frames), dtype=np.float64)

    y_oct = np.asarray(y, dtype=np.float64)
    for j in range(n_octaves):
        hop_j = hop_length // (2 ** j)
        if j > 0:
            # One octave down == same kernels at half the sample rate.
            y_oct = resample_poly(y_oct, 1, 2).astype(np.float64)

        if y_oct.size < 2:
            break

        Y = _frame_fft(y_oct, n_fft, hop_j, n_frames)
        # (n_frames, n_fft) @ (n_fft, bins) -> (n_frames, bins)
        resp = (Y @ K_conj.T) / n_fft

        # Octave j counts down from the top of the bin range.
        lo = (n_octaves - 1 - j) * bins_per_octave
        out[lo:lo + bins_per_octave, :] = np.abs(resp).T

    spec = CQTSpec(
        sample_rate=sr,
        f_min=f_min,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        hop_length=hop_length,
        n_octaves=n_octaves,
        quality_factor=quality_factor,
        frame_rate_hz=sr / hop_length,
    )
    return out, spec


def harmonic_percussive_split(
    S: np.ndarray,
    kernel_harmonic: int = 17,
    kernel_percussive: int = 17,
    power: float = 2.0,
    mask_exponent: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate a magnitude spectrogram into harmonic and percussive parts.

    Harmonic content is smooth along *time* (a held note keeps the same
    frequency); percussive content is smooth along *frequency* (a drum hit
    spreads energy across the whole spectrum at one instant).  Median
    filtering along each axis and then building soft masks is the standard
    Fitzgerald method.  Removing drums before chroma extraction measurably
    reduces spurious chord changes on the beat.
    """
    from scipy.ndimage import median_filter

    H = median_filter(S, size=(1, kernel_harmonic), mode="nearest")
    P = median_filter(S, size=(kernel_percussive, 1), mode="nearest")

    Hp = np.power(H, power)
    Pp = np.power(P, power)
    denom = Hp + Pp + 1e-12

    mask_h = np.power(Hp / denom, mask_exponent)
    mask_p = np.power(Pp / denom, mask_exponent)
    return S * mask_h, S * mask_p
