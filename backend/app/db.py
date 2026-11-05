"""Database models and session handling.

The schema follows the one in the specification exactly, with the field
names it asked for.  Times are stored as integer milliseconds throughout
so that words and chords share one timeline with no float drift and no
unit conversion anywhere in the stack.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import (
    create_engine, String, Integer, Float, Boolean, DateTime, ForeignKey,
    Text, Index, JSON, event,
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker, Session,
)

from .config import get_settings


def _uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ProjectStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    ANALYZING = "analyzing"
    READY = "ready"
    FAILED = "failed"


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TimingSource(str, enum.Enum):
    """Where a word's timing came from.

    This exists because the specification is explicit that estimated
    times must not be presented as if they were measured.  Every word
    carries its provenance, the API returns it, and the UI shows
    estimated timings differently from measured ones.
    """
    PROVIDER_WORD = "provider_word"       # the provider timed each word
    FORCED_ALIGNMENT = "forced_alignment" # we aligned within a timed phrase
    ESTIMATED = "estimated"               # proportional fallback
    USER = "user"                         # a human set it


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[Optional[str]] = mapped_column(String(320), unique=True, nullable=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(120))
    password_hash: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    projects: Mapped[List["Project"]] = relationship(back_populates="user",
                                                     cascade="all, delete-orphan")
    settings: Mapped[Optional["UserSettings"]] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan")


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(300))
    file_name: Mapped[str] = mapped_column(String(300))
    storage_key: Mapped[str] = mapped_column(String(400))
    content_type: Mapped[Optional[str]] = mapped_column(String(120))
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default=ProjectStatus.UPLOADED.value, index=True)

    # Analysis output that belongs to the project as a whole.
    key_tonic: Mapped[Optional[str]] = mapped_column(String(4))
    key_mode: Mapped[Optional[str]] = mapped_column(String(10))
    key_confidence: Mapped[Optional[float]] = mapped_column(Float)
    tempo_bpm: Mapped[Optional[float]] = mapped_column(Float)
    tuning_offset_cents: Mapped[Optional[float]] = mapped_column(Float)
    beat_times_ms: Mapped[Optional[list]] = mapped_column(JSON)
    onset_times_ms: Mapped[Optional[list]] = mapped_column(JSON)
    transpose_semitones: Mapped[int] = mapped_column(Integer, default=0)
    waveform_peaks: Mapped[Optional[list]] = mapped_column(JSON)
    analysis_diagnostics: Mapped[Optional[dict]] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=utcnow, onupdate=utcnow)

    user: Mapped["User"] = relationship(back_populates="projects")
    words: Mapped[List["TranscriptWord"]] = relationship(
        back_populates="project", cascade="all, delete-orphan",
        order_by="TranscriptWord.start_time_ms")
    chords: Mapped[List["ChordEvent"]] = relationship(
        back_populates="project", cascade="all, delete-orphan",
        order_by="ChordEvent.start_time_ms")
    jobs: Mapped[List["AnalysisJob"]] = relationship(
        back_populates="project", cascade="all, delete-orphan",
        order_by="AnalysisJob.created_at")


class TranscriptWord(Base):
    __tablename__ = "transcript_words"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    text: Mapped[str] = mapped_column(String(200))
    start_time_ms: Mapped[int] = mapped_column(Integer)
    end_time_ms: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    line_id: Mapped[int] = mapped_column(Integer, default=0, index=True)
    word_index: Mapped[int] = mapped_column(Integer, default=0)
    timing_source: Mapped[str] = mapped_column(String(24),
                                               default=TimingSource.PROVIDER_WORD.value)
    is_edited: Mapped[bool] = mapped_column(Boolean, default=False)
    language: Mapped[Optional[str]] = mapped_column(String(12))

    project: Mapped["Project"] = relationship(back_populates="words")


class ChordEvent(Base):
    __tablename__ = "chord_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    chord_name: Mapped[str] = mapped_column(String(24))
    root: Mapped[str] = mapped_column(String(4))
    quality: Mapped[str] = mapped_column(String(12))
    bass_note: Mapped[Optional[str]] = mapped_column(String(4))
    start_time_ms: Mapped[int] = mapped_column(Integer)
    end_time_ms: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    alternatives: Mapped[Optional[list]] = mapped_column(JSON)
    is_edited: Mapped[bool] = mapped_column(Boolean, default=False)
    detection_version: Mapped[str] = mapped_column(String(16), default="1.0.0")

    project: Mapped["Project"] = relationship(back_populates="chords")


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    stage: Mapped[str] = mapped_column(String(40), default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.QUEUED.value, index=True)
    message: Mapped[Optional[str]] = mapped_column(String(400))
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    stages: Mapped[Optional[list]] = mapped_column(JSON)
    want_transcription: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    project: Mapped["Project"] = relationship(back_populates="jobs")


class UserSettings(Base):
    __tablename__ = "user_settings"
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    transcription_provider: Mapped[Optional[str]] = mapped_column(String(40))
    encrypted_api_key: Mapped[Optional[str]] = mapped_column(Text)
    """Ciphertext only. The plaintext key is never stored, never logged and
    never returned by any endpoint."""
    api_key_hint: Mapped[Optional[str]] = mapped_column(String(16))
    api_key_updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    preferred_language: Mapped[Optional[str]] = mapped_column(String(12))
    theme: Mapped[str] = mapped_column(String(10), default="dark")
    chord_notation: Mapped[str] = mapped_column(String(10), default="sharps")
    allow_audio_upload: Mapped[bool] = mapped_column(Boolean, default=False)

    user: Mapped["User"] = relationship(back_populates="settings")


Index("ix_words_project_line", TranscriptWord.project_id, TranscriptWord.line_id,
      TranscriptWord.word_index)
Index("ix_chords_project_start", ChordEvent.project_id, ChordEvent.start_time_ms)


_settings = get_settings()
_connect_args = {"check_same_thread": False} if _settings.database_url.startswith("sqlite") else {}

engine = create_engine(
    _settings.database_url,
    connect_args=_connect_args,
    pool_pre_ping=True,
    future=True,
)

if _settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record):
        cur = dbapi_connection.cursor()
        # WAL lets the analysis worker write progress while the API reads.
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False,
                            class_=Session, future=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_session():
    """FastAPI dependency: one session per request."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
