"""Production adapter for the replacement chord recognizer.

The recognition algorithm lives in :mod:`recognizer_v2`. This adapter keeps
ChordVision's existing API/database contract stable, so the player, lyric
alignment, editor, exports and saved projects do not need a front-end rewrite.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import time

import numpy as np

from .audio import decode as decode_audio, probe as probe_audio
from .theory import parse_chord, pc_name, NO_CHORD
from .recognizer_v2 import analyze as run_recognizer

ProgressFn = Callable[[str, float, str], None]
ENGINE_VERSION = "2.0.0"


@dataclass
class EngineConfig:
    """Settings for the replacement recognizer."""
    quality_set: str = "sevenths"
    sample_rate: int = 22050
    max_duration_s: Optional[float] = 300.0
    mode: str = "nnls"
    slash_chords: bool = True
    min_segment_s: float = 1.0
    refine_boundaries: bool = True
    snap_to_onsets: bool = False


class WireResult:
    """Small wrapper preserving the existing ``result.to_dict()`` call site."""
    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload

    def to_dict(self) -> Dict[str, Any]:
        return self._payload


class ChordEngine:
    """Analyze uploaded audio using the supplied replacement engine."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()

    def analyze_file(self, path: str, progress: Optional[ProgressFn] = None) -> WireResult:
        cfg = self.config
        started = time.perf_counter()
        self._emit(progress, "prepare_audio", 0.0, "Checking the audio file")
        source = probe_audio(path)
        samples, sr = decode_audio(
            path,
            target_sr=cfg.sample_rate,
            mono=True,
            max_seconds=cfg.max_duration_s,
        )
        self._emit(progress, "prepare_audio", 1.0, "Audio decoded and ready for analysis")

        result = run_recognizer(
            samples=samples,
            sr=sr,
            max_seconds=cfg.max_duration_s,
            vocab=self._vocabulary(cfg.quality_set),
            mode=cfg.mode,
            slash=cfg.slash_chords,
            min_segment=cfg.min_segment_s,
            refine=cfg.refine_boundaries,
            snap_onsets=cfg.snap_to_onsets,
            progress=progress,
        )
        elapsed = time.perf_counter() - started
        return WireResult(self._to_chordvision_payload(result, source, cfg, elapsed))

    @staticmethod
    def _emit(progress: Optional[ProgressFn], stage: str, fraction: float, message: str) -> None:
        if progress is not None:
            progress(stage, float(np.clip(fraction, 0.0, 1.0)), message)

    @staticmethod
    def _vocabulary(quality_set: str) -> str:
        # These names are the existing ChordVision settings. The new engine
        # implements the same useful sets (including m7b5) under these names.
        return quality_set if quality_set in {
            "triads", "triads_ext", "standard", "sevenths", "full"
        } else "sevenths"

    @staticmethod
    def _to_chordvision_payload(result, source, cfg: EngineConfig,
                                elapsed_s: float) -> Dict[str, Any]:
        chords = []
        duration_ms = int(round(result.duration * 1000.0))

        for segment in result.chords:
            name = str(segment.chord)
            start_ms = max(0, min(duration_ms, int(round(segment.start * 1000.0))))
            end_ms = max(start_ms, min(duration_ms, int(round(segment.end * 1000.0))))

            if name == NO_CHORD:
                root, quality, bass_note = NO_CHORD, NO_CHORD, None
            else:
                root_pc, quality, bass_pc = parse_chord(name)
                root = pc_name(root_pc)
                bass_note = pc_name(bass_pc) if bass_pc is not None else None

            alternatives = []
            for alternative in (segment.alternatives or []):
                alt_name = str(alternative.get("chord", ""))
                if not alt_name or alt_name == NO_CHORD:
                    continue
                alternatives.append({
                    "chordName": alt_name,
                    "confidence": round(float(np.clip(alternative.get("confidence", 0.0), 0.0, 1.0)), 4),
                })

            chords.append({
                "chordName": name,
                "root": root,
                "quality": quality,
                "bassNote": bass_note,
                "startTimeMs": start_ms,
                "endTimeMs": end_ms,
                "confidence": round(float(np.clip(segment.confidence, 0.0, 1.0)), 4),
                "alternatives": alternatives,
                "isEdited": False,
                "detectionVersion": ENGINE_VERSION,
            })

        key_parts = str(result.key).rsplit(" ", 1)
        key_tonic = key_parts[0] if len(key_parts) == 2 else None
        key_mode = key_parts[1] if len(key_parts) == 2 else None
        beat_times_ms = [
            max(0, min(duration_ms, int(round(float(t) * 1000.0))))
            for t in (result.beats or [])
        ]
        onset_times_ms = [
            max(0, min(duration_ms, int(round(float(t) * 1000.0))))
            for t in (result.onsets or [])
        ]
        frame_rate_hz = cfg.sample_rate / 512.0

        return {
            "chords": chords,
            "key": {
                "tonic": key_tonic,
                "mode": key_mode,
                # The supplied recognizer estimates a key but does not
                # expose a calibrated key-confidence score.
                "confidence": None,
                "alternatives": [],
            },
            "durationMs": duration_ms,
            "sampleRate": cfg.sample_rate,
            "tuningOffsetCents": round(float(result.tuning_cents), 2),
            "beatTimesMs": beat_times_ms,
            "onsetTimesMs": onset_times_ms,
            "tempoBpm": round(float(result.tempo), 2) if result.tempo else None,
            "frameRateHz": round(frame_rate_hz, 4),
            "detectionVersion": ENGINE_VERSION,
            "diagnostics": {
                "engineVersion": ENGINE_VERSION,
                "recognizer": "harmonic-cqt-nnls-viterbi",
                "qualitySet": cfg.quality_set,
                "analysisMode": cfg.mode,
                "chordCount": len([c for c in chords if c["chordName"] != NO_CHORD]),
                "confidenceMethod": "mean segment emission probabilities",
                "analysisSeconds": round(elapsed_s, 3),
                "realtimeFactor": round((duration_ms / 1000.0) / elapsed_s, 2) if elapsed_s > 0 else None,
                "source": {
                    "container": source.format_name,
                    "codec": source.codec_name,
                    "originalSampleRate": source.sample_rate,
                    "originalChannels": source.channels,
                    "probedDurationS": round(source.duration_s, 3),
                },
            },
        }
