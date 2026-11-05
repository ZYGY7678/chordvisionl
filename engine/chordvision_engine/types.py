"""Data types shared across the engine."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class ChordCandidate:
    """One alternative reading of a chord segment."""
    chord_name: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"chordName": self.chord_name, "confidence": round(self.confidence, 4)}


@dataclass
class ChordEvent:
    """A chord that is sounding over a contiguous span of the audio.

    Times are in milliseconds on the same timeline as the audio file and
    as the transcript words, so chord and word events can be interleaved
    without any rescaling.
    """
    chord_name: str          # display name, e.g. "Am", "C/E", "Gmaj7"
    root: str                # pitch class name of the root, e.g. "A"
    quality: str             # "maj", "min", "7", "maj7", ...
    bass_note: Optional[str] # pitch class of the bass if it is not the root
    start_time_ms: int
    end_time_ms: int
    confidence: float
    alternatives: List[ChordCandidate] = field(default_factory=list)
    is_edited: bool = False
    detection_version: str = "1.0.0"

    @property
    def duration_ms(self) -> int:
        return self.end_time_ms - self.start_time_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chordName": self.chord_name,
            "root": self.root,
            "quality": self.quality,
            "bassNote": self.bass_note,
            "startTimeMs": self.start_time_ms,
            "endTimeMs": self.end_time_ms,
            "confidence": round(self.confidence, 4),
            "alternatives": [a.to_dict() for a in self.alternatives],
            "isEdited": self.is_edited,
            "detectionVersion": self.detection_version,
        }


@dataclass
class KeyEstimate:
    """Global key of the piece."""
    tonic: str                  # "C", "F#", ...
    mode: str                   # "major" | "minor"
    confidence: float
    alternatives: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.tonic} {self.mode}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tonic": self.tonic,
            "mode": self.mode,
            "name": self.name,
            "confidence": round(self.confidence, 4),
            "alternatives": self.alternatives,
        }


@dataclass
class Segment:
    """A low-level frame span used internally before chord events are built."""
    start_frame: int
    end_frame: int
    state: int


@dataclass
class AnalysisResult:
    """Everything the engine produces for one audio file."""
    chords: List[ChordEvent]
    key: KeyEstimate
    duration_ms: int
    sample_rate: int
    tuning_offset_cents: float
    beat_times_ms: List[int]
    onset_times_ms: List[int]
    tempo_bpm: Optional[float]
    frame_rate_hz: float
    detection_version: str = "1.0.0"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chords": [c.to_dict() for c in self.chords],
            "key": self.key.to_dict(),
            "durationMs": self.duration_ms,
            "sampleRate": self.sample_rate,
            "tuningOffsetCents": round(self.tuning_offset_cents, 2),
            "beatTimesMs": self.beat_times_ms,
            "onsetTimesMs": self.onset_times_ms,
            "tempoBpm": round(self.tempo_bpm, 2) if self.tempo_bpm else None,
            "frameRateHz": round(self.frame_rate_hz, 4),
            "detectionVersion": self.detection_version,
            "diagnostics": self.diagnostics,
        }
