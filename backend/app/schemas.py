"""Request and response shapes.

Field names are camelCase on the wire to match the specification's data
model and what the frontend expects, while staying snake_case in Python.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator


def _camel(s: str) -> str:
    head, *rest = s.split("_")
    return head + "".join(w.capitalize() for w in rest)


class Wire(BaseModel):
    model_config = ConfigDict(alias_generator=_camel, populate_by_name=True,
                              from_attributes=True)


class ChordOut(Wire):
    id: str
    chord_name: str
    root: str
    quality: str
    bass_note: Optional[str] = None
    start_time_ms: int
    end_time_ms: int
    confidence: float
    alternatives: Optional[List[Dict[str, Any]]] = None
    is_edited: bool = False
    detection_version: str = "1.0.0"


class WordOut(Wire):
    id: str
    text: str
    start_time_ms: int
    end_time_ms: int
    confidence: Optional[float] = None
    line_id: int
    word_index: int
    timing_source: str
    is_edited: bool = False
    language: Optional[str] = None


class ProjectSummary(Wire):
    id: str
    title: str
    file_name: str
    duration_ms: int
    file_size: int
    status: str
    key_tonic: Optional[str] = None
    key_mode: Optional[str] = None
    tempo_bpm: Optional[float] = None
    transpose_semitones: int = 0
    created_at: Optional[str] = None
    has_lyrics: bool = False
    chord_count: int = 0


class ProjectDetail(ProjectSummary):
    key_confidence: Optional[float] = None
    tuning_offset_cents: Optional[float] = None
    beat_times_ms: Optional[List[int]] = None
    onset_times_ms: Optional[List[int]] = None
    waveform_peaks: Optional[List[List[float]]] = None
    analysis_diagnostics: Optional[Dict[str, Any]] = None
    chords: List[ChordOut] = Field(default_factory=list)
    words: List[WordOut] = Field(default_factory=list)


class JobOut(Wire):
    id: str
    project_id: str
    stage: str
    progress: float
    status: str
    message: Optional[str] = None
    error_message: Optional[str] = None
    stages: Optional[List[Dict[str, Any]]] = None


class SettingsOut(Wire):
    transcription_provider: Optional[str] = None
    api_key_hint: Optional[str] = None
    api_key_set: bool = False
    api_key_updated_at: Optional[str] = None
    preferred_language: Optional[str] = None
    theme: str = "dark"
    chord_notation: str = "sharps"
    allow_audio_upload: bool = False


class SettingsIn(Wire):
    transcription_provider: Optional[str] = None
    api_key: Optional[str] = None
    """Write-only. It is encrypted on arrival and never returned."""
    preferred_language: Optional[str] = None
    theme: Optional[str] = None
    chord_notation: Optional[str] = None
    allow_audio_upload: Optional[bool] = None

    @field_validator("theme")
    @classmethod
    def _theme(cls, v):
        if v is not None and v not in ("dark", "light", "system"):
            raise ValueError("theme must be dark, light or system")
        return v

    @field_validator("chord_notation")
    @classmethod
    def _notation(cls, v):
        if v is not None and v not in ("sharps", "flats"):
            raise ValueError("chordNotation must be sharps or flats")
        return v


class VerifyKeyIn(Wire):
    provider: str
    api_key: str


class VerifyKeyOut(Wire):
    valid: bool
    message: str


class AnalyzeIn(Wire):
    transcribe: bool = False


class ProjectPatch(Wire):
    title: Optional[str] = None
    transpose_semitones: Optional[int] = None

    @field_validator("transpose_semitones")
    @classmethod
    def _range(cls, v):
        if v is not None and not (-12 <= v <= 12):
            raise ValueError("transposeSemitones must be between -12 and 12")
        return v


class ChordPatch(Wire):
    chord_name: Optional[str] = None
    start_time_ms: Optional[int] = None
    end_time_ms: Optional[int] = None


class ChordCreate(Wire):
    chord_name: str
    start_time_ms: int
    end_time_ms: int


class WordPatch(Wire):
    text: Optional[str] = None
    start_time_ms: Optional[int] = None
    end_time_ms: Optional[int] = None
    line_id: Optional[int] = None


class WordCreate(Wire):
    text: str
    start_time_ms: int
    end_time_ms: int
    line_id: int = 0


class WordSplit(Wire):
    at_ratio: float = 0.5
    first_text: Optional[str] = None
    second_text: Optional[str] = None
