"""The transcription provider interface.

Transcription is the one part of the system that is deliberately *not*
implemented here: the specification requires it to go through an external
API with a key the user supplies.  What this module provides is the shape
that every provider has to fit, so that swapping provider is a
configuration change rather than a code change.

Every adapter must answer the same question: for this audio, what words
were said and when.  Providers differ in how much of that they give:

  * some return a start and end time for every word;
  * some return times only per phrase or per segment;
  * some return no times at all.

The adapter reports which of those it got, per word, in ``timing_source``.
Nothing downstream is allowed to treat an estimate as a measurement, and
that field is how the rest of the system knows the difference.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Protocol

from ..db import TimingSource


class TranscriptionError(RuntimeError):
    """A provider call failed. The message is safe to show the user."""

    def __init__(self, message: str, *, status_code: Optional[int] = None,
                 retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


@dataclass
class Word:
    text: str
    start_ms: int
    end_ms: int
    confidence: Optional[float] = None
    timing_source: str = TimingSource.PROVIDER_WORD.value


@dataclass
class Segment:
    """A phrase or sentence, as the provider delimited it."""
    text: str
    start_ms: int
    end_ms: int
    words: List[Word] = field(default_factory=list)
    confidence: Optional[float] = None


@dataclass
class TranscriptionResult:
    provider: str
    language: Optional[str]
    text: str
    segments: List[Segment]
    words: List[Word]
    has_word_timings: bool
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "language": self.language,
            "text": self.text,
            "hasWordTimings": self.has_word_timings,
            "words": [
                {"text": w.text, "startTimeMs": w.start_ms, "endTimeMs": w.end_ms,
                 "confidence": w.confidence, "timingSource": w.timing_source}
                for w in self.words
            ],
        }


@dataclass
class ProviderInfo:
    """What the settings screen needs in order to describe a provider."""
    id: str
    name: str
    docs_url: str
    key_format_hint: str
    gives_word_timings: bool
    notes: str = ""


class TranscriptionProvider(abc.ABC):
    """Base class for every transcription adapter."""

    info: ProviderInfo

    def __init__(self, api_key: str, timeout_s: float = 600.0):
        if not api_key:
            raise TranscriptionError("No API key was supplied for this provider.")
        self._api_key = api_key
        self._timeout_s = timeout_s

    @abc.abstractmethod
    async def transcribe(self, audio_path: str,
                         language: Optional[str] = None) -> TranscriptionResult:
        """Transcribe one audio file."""

    @abc.abstractmethod
    async def verify_key(self) -> bool:
        """Check the key works, without transcribing anything.

        The settings screen calls this so the user finds out their key is
        wrong immediately, rather than after uploading a song and waiting
        for a job to fail.
        """

    # -- shared helpers --------------------------------------------------

    @staticmethod
    def _ms(seconds: Optional[float]) -> int:
        return int(round((seconds or 0.0) * 1000.0))

    def _safe_error(self, exc: Exception) -> str:
        """Error text with the API key scrubbed out of it."""
        from ..crypto import redact
        return redact(str(exc), self._api_key)
