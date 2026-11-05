"""Focused smoke test for the production replacement chord recognizer."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "engine"))

from chordvision_engine.engine_adapter import ChordEngine, EngineConfig  # noqa: E402
from chordvision_engine.recognizer_v2 import VOCABULARIES  # noqa: E402


def _note(midi: int, seconds: float, sample_rate: int, cents: float,
          amplitude: float = 0.15) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    frequency = 440.0 * 2.0 ** ((midi - 69 + cents / 100.0) / 12.0)
    signal = np.zeros_like(t)
    for harmonic in range(1, 7):
        signal += (amplitude / harmonic) * np.sin(
            2.0 * np.pi * frequency * harmonic * t
        )
    return signal * (0.4 + 0.6 * np.exp(-0.8 * t))


def main() -> int:
    assert "m7b5" in VOCABULARIES["sevenths"], "seventh vocabulary lost m7b5"

    sample_rate = 22050
    chord_seconds = 1.8
    tuning_cents = 35.0
    progression = [
        (60, [0, 4, 7]),  # C
        (67, [0, 4, 7]),  # G
        (69, [0, 3, 7]),  # Am
        (65, [0, 4, 7]),  # F
    ]
    audio_parts = []
    for root, intervals in progression:
        chord = sum(
            (_note(root + interval, chord_seconds, sample_rate, tuning_cents)
             for interval in intervals),
            np.zeros(int(chord_seconds * sample_rate), dtype=np.float32),
        )
        bass = _note(root - 24, chord_seconds, sample_rate, tuning_cents, 0.35)
        audio_parts.append(chord + bass)
    audio = np.concatenate(audio_parts)

    with tempfile.TemporaryDirectory(prefix="chordvision-v2-test-") as tmp:
        audio_path = os.path.join(tmp, "detuned-progression.wav")
        sf.write(audio_path, audio, sample_rate)
        payload = ChordEngine(
            EngineConfig(quality_set="sevenths", max_duration_s=30.0)
        ).analyze_file(audio_path).to_dict()

    detected_roots = [
        chord["root"] for chord in payload["chords"]
        if chord["chordName"] != "N"
    ]
    assert detected_roots == ["C", "G", "A", "F"], detected_roots
    assert 25.0 <= payload["tuningOffsetCents"] <= 45.0, payload["tuningOffsetCents"]
    assert payload["key"]["tonic"] == "C", payload["key"]
    assert payload["diagnostics"]["engineVersion"] == "2.0.0"
    assert all(0.0 <= chord["confidence"] <= 1.0 for chord in payload["chords"])
    print("PASS  replacement engine detects C–G–Am–F after +35-cent detuning")
    print(f"PASS  measured tuning correction: {payload['tuningOffsetCents']:.1f} cents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
