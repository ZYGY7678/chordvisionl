"""Regression tests for block-wise analysis on memory-limited instances."""
from __future__ import annotations

import os
import sys

import librosa
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "engine"))

from chordvision_engine.recognizer_v2 import (  # noqa: E402
    _harmonic_chunked,
    _cqt_chunked,
    _onset_strength_chunked,
    _frame_rms_chunked,
)


def main() -> int:
    sr = 22050
    hop = 512
    seconds = 29.5
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    signal = (
        0.22 * np.sin(2 * np.pi * 130.81 * t)
        + 0.14 * np.sin(2 * np.pi * 164.81 * t)
        + 0.08 * np.sin(2 * np.pi * 196.00 * t)
    ).astype(np.float32)
    signal[::sr // 2] += 0.18

    hp_progress = []
    harmonic = _harmonic_chunked(
        signal, 3.0, sr, progress=lambda *event: hp_progress.append(event)
    )
    assert harmonic.shape == signal.shape
    assert np.isfinite(harmonic).all()
    assert len(hp_progress) > 1

    cqt_progress = []
    chunked_cqt = _cqt_chunked(
        harmonic, sr, hop, 0.0,
        progress=lambda *event: cqt_progress.append(event),
    )
    reference_cqt = np.abs(librosa.cqt(
        harmonic,
        sr=sr,
        hop_length=hop,
        fmin=librosa.note_to_hz("C2"),
        n_bins=216,
        bins_per_octave=36,
        tuning=0.0,
    )).astype(np.float32)
    assert chunked_cqt.shape == reference_cqt.shape
    assert np.allclose(chunked_cqt, reference_cqt, rtol=1e-4, atol=1e-5)
    assert len(cqt_progress) > 1

    onset_progress = []
    onset = _onset_strength_chunked(
        signal, sr, hop, progress=lambda *event: onset_progress.append(event)
    )
    rms = _frame_rms_chunked(harmonic, hop)
    expected_frames = len(signal) // hop + 1
    assert onset.shape == (expected_frames,)
    assert rms.shape == (expected_frames,)
    assert np.isfinite(onset).all() and np.isfinite(rms).all()
    assert len(onset_progress) > 1

    print("PASS  HPSS, CQT, onset and RMS operate in bounded blocks")
    print("PASS  chunked CQT matches the whole-signal reference within tolerance")
    print("PASS  progress is emitted while long stages are still running")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())