"""ChordVision HTTP API."""
from __future__ import annotations

import asyncio
import json
import os
import mimetypes
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from fastapi import (
    FastAPI, Depends, HTTPException, UploadFile, File, Form, Request,
    Response, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from .config import get_settings
from .db import (
    init_db, get_session, User, Project, ChordEvent, TranscriptWord,
    AnalysisJob, UserSettings, ProjectStatus, JobStatus, TimingSource, utcnow,
)
from .crypto import encrypt_api_key, decrypt_api_key, key_hint, DecryptionError
from .security import get_current_user, rate_limit, upload_rate_limit
from . import storage
from .jobs import get_queue, shutdown_queue, recover_interrupted_jobs, STAGES
from .transcription import get_provider, list_providers, TranscriptionError
from . import schemas as S
from .sync import synchronize
from . import export as exporters

import sys
ENGINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "engine")
if ENGINE_PATH not in sys.path:
    sys.path.insert(0, ENGINE_PATH)
from chordvision_engine.theory import (  # noqa: E402
    transpose_chord, parse_chord, chord_name as build_chord_name, pc_name,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # In production a missing secret must stop the server here, not on the
    # first request that happens to need it.
    get_settings().resolved_secret()
    init_db()
    os.makedirs(os.path.abspath(get_settings().storage_dir), exist_ok=True)
    recover_interrupted_jobs()
    get_queue()
    yield
    shutdown_queue()


settings = get_settings()
app = FastAPI(
    title="ChordVision API",
    version="1.0.0",
    description="Chord recognition, lyric transcription and synchronisation.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- helpers ----------------------------------------------------------------

def _owned_project(project_id: str, user: User, session: Session) -> Project:
    project = session.get(Project, project_id)
    if project is None or project.user_id != user.id:
        # The same answer whether it does not exist or belongs to someone
        # else, so project ids cannot be probed for existence.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found.")
    return project


def _iso(dt: Optional[datetime]) -> Optional[str]:
    """Serialise a timestamp with an explicit UTC offset.

    SQLite does not keep tzinfo, so a value written as aware UTC reads
    back naive. Serialised without an offset, the browser then parses it
    as *local* time and a project created a minute ago is labelled hours
    old. Reattaching UTC here is what keeps "just now" saying just now.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _notation_flats(session: Session, user_id: str) -> bool:
    us = session.get(UserSettings, user_id)
    return bool(us and us.chord_notation == "flats")


def _chord_out(c: ChordEvent, semitones: int = 0, use_flats: bool = False) -> Dict[str, Any]:
    name = c.chord_name
    root = c.root
    bass = c.bass_note
    if semitones or use_flats:
        name = transpose_chord(c.chord_name, semitones, use_flats)
        r, _q, b = parse_chord(name)
        root = pc_name(r, use_flats)
        bass = pc_name(b, use_flats) if b is not None else None
    return {
        "id": c.id, "chordName": name, "root": root, "quality": c.quality,
        "bassNote": bass, "startTimeMs": c.start_time_ms,
        "endTimeMs": c.end_time_ms, "confidence": round(c.confidence, 4),
        "alternatives": [
            {**a, "chordName": transpose_chord(a["chordName"], semitones, use_flats)}
            if semitones or use_flats else a
            for a in (c.alternatives or [])
        ],
        "isEdited": c.is_edited, "detectionVersion": c.detection_version,
    }


def _word_out(w: TranscriptWord) -> Dict[str, Any]:
    return {
        "id": w.id, "text": w.text, "startTimeMs": w.start_time_ms,
        "endTimeMs": w.end_time_ms, "confidence": w.confidence,
        "lineId": w.line_id, "wordIndex": w.word_index,
        "timingSource": w.timing_source, "isEdited": w.is_edited,
        "language": w.language,
    }


def _project_summary(p: Project, session: Session) -> Dict[str, Any]:
    chord_count = session.scalar(
        select(func.count()).select_from(ChordEvent).where(ChordEvent.project_id == p.id)) or 0
    word_count = session.scalar(
        select(func.count()).select_from(TranscriptWord).where(TranscriptWord.project_id == p.id)) or 0
    return {
        "id": p.id, "title": p.title, "fileName": p.file_name,
        "durationMs": p.duration_ms, "fileSize": p.file_size, "status": p.status,
        "keyTonic": p.key_tonic, "keyMode": p.key_mode, "tempoBpm": p.tempo_bpm,
        "transposeSemitones": p.transpose_semitones,
        "createdAt": _iso(p.created_at), "hasLyrics": word_count > 0,
        "chordCount": chord_count,
    }


# --- meta -------------------------------------------------------------------

@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": app.version,
        "engineVersion": __import__("chordvision_engine").__version__,
        "stages": STAGES,
        "limits": {
            "maxUploadBytes": settings.max_upload_bytes,
            "maxAudioSeconds": settings.max_audio_seconds,
        },
        "supportedFormats": sorted(e.lstrip(".") for e in storage.ALLOWED_EXTENSIONS),
        "persistence": {
            "database": "postgres" if get_settings().database_url.startswith(("postgresql://", "postgresql+")) else "temporary",
            "audioStorage": "supabase" if storage.remote_storage_ready() else "temporary",
            "ready": get_settings().database_url.startswith(("postgresql://", "postgresql+")) and storage.remote_storage_ready(),
        },
    }


@app.get("/api/providers")
def providers(_: None = Depends(rate_limit)) -> List[Dict[str, Any]]:
    return list_providers()


# --- settings ----------------------------------------------------------------

@app.get("/api/settings")
def read_settings(user: User = Depends(get_current_user),
                  session: Session = Depends(get_session)) -> Dict[str, Any]:
    us = session.get(UserSettings, user.id)
    if us is None:
        us = UserSettings(user_id=user.id)
        session.add(us)
        session.commit()
    return {
        "transcriptionProvider": us.transcription_provider,
        "apiKeyHint": us.api_key_hint,
        "apiKeySet": bool(us.encrypted_api_key),
        "apiKeyUpdatedAt": _iso(us.api_key_updated_at),
        "preferredLanguage": us.preferred_language,
        "theme": us.theme,
        "chordNotation": us.chord_notation,
        "allowAudioUpload": us.allow_audio_upload,
    }


@app.put("/api/settings")
def write_settings(payload: S.SettingsIn,
                   user: User = Depends(get_current_user),
                   session: Session = Depends(get_session),
                   _: None = Depends(rate_limit)) -> Dict[str, Any]:
    us = session.get(UserSettings, user.id)
    if us is None:
        us = UserSettings(user_id=user.id)
        session.add(us)

    if payload.transcription_provider is not None:
        from .transcription import PROVIDERS
        if payload.transcription_provider not in PROVIDERS:
            raise HTTPException(400, f"Unknown provider: {payload.transcription_provider}")
        us.transcription_provider = payload.transcription_provider

    if payload.api_key:
        # Encrypted immediately. The plaintext exists only in this frame.
        us.encrypted_api_key = encrypt_api_key(payload.api_key)
        us.api_key_hint = key_hint(payload.api_key)
        us.api_key_updated_at = utcnow()

    for field in ("preferred_language", "theme", "chord_notation", "allow_audio_upload"):
        value = getattr(payload, field)
        if value is not None:
            setattr(us, field, value)

    session.commit()
    return read_settings(user, session)


@app.delete("/api/settings/api-key")
def delete_api_key(user: User = Depends(get_current_user),
                   session: Session = Depends(get_session)) -> Dict[str, Any]:
    us = session.get(UserSettings, user.id)
    if us:
        us.encrypted_api_key = None
        us.api_key_hint = None
        us.api_key_updated_at = None
        session.commit()
    return {"ok": True}


@app.post("/api/settings/verify-key")
async def verify_key(payload: S.VerifyKeyIn,
                     _: None = Depends(rate_limit)) -> Dict[str, Any]:
    """Check a key works before the user commits to it."""
    try:
        provider = get_provider(payload.provider, payload.api_key,
                                settings.transcription_timeout_s)
        ok = await provider.verify_key()
    except TranscriptionError as exc:
        return {"valid": False, "message": str(exc)}
    except Exception as exc:
        return {"valid": False, "message": f"Could not check the key: {exc}"}
    return {
        "valid": ok,
        "message": "The key works." if ok else "The provider rejected this key.",
    }


# --- projects -----------------------------------------------------------------

@app.post("/api/projects", status_code=status.HTTP_201_CREATED)
def create_project(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    transcribe: bool = Form(False),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    _rl: None = Depends(rate_limit),
    _url: None = Depends(upload_rate_limit),
) -> Dict[str, Any]:
    try:
        stored = storage.save_upload(file.file, file.filename or "audio",
                                     file.content_type)
    except storage.UploadRejected as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    if transcribe:
        us = session.get(UserSettings, user.id)
        if us is None or not us.encrypted_api_key:
            storage.delete(stored.storage_key)
            raise HTTPException(
                400, "Transcription was requested but no API key is configured. "
                     "Add one in settings first.")
        if not us.allow_audio_upload:
            storage.delete(stored.storage_key)
            raise HTTPException(
                400, "Transcription sends this audio to a third-party provider. "
                     "Turn that on in settings first.")

    base = os.path.splitext(file.filename or "Untitled")[0]
    project = Project(
        user_id=user.id,
        title=(title or base).strip()[:300] or "Untitled",
        file_name=file.filename or "audio",
        storage_key=stored.storage_key,
        content_type=stored.content_type,
        duration_ms=stored.duration_ms,
        file_size=stored.size_bytes,
        status=ProjectStatus.UPLOADED.value,
    )
    session.add(project)
    session.flush()

    job = AnalysisJob(project_id=project.id, stage="upload", progress=0.02,
                      status=JobStatus.QUEUED.value, stages=STAGES,
                      want_transcription=bool(transcribe),
                      message="Queued for analysis")
    session.add(job)
    session.commit()

    get_queue().submit(job.id, project.id)

    return {"project": _project_summary(project, session), "jobId": job.id}


@app.get("/api/projects")
def list_projects(limit: int = 50,
                  user: User = Depends(get_current_user),
                  session: Session = Depends(get_session)) -> List[Dict[str, Any]]:
    rows = session.scalars(
        select(Project).where(Project.user_id == user.id)
        .order_by(Project.created_at.desc()).limit(min(max(limit, 1), 200))
    ).all()
    return [_project_summary(p, session) for p in rows]


@app.get("/api/projects/{project_id}")
def get_project(project_id: str,
                user: User = Depends(get_current_user),
                session: Session = Depends(get_session)) -> Dict[str, Any]:
    p = _owned_project(project_id, user, session)
    flats = _notation_flats(session, user.id)
    out = _project_summary(p, session)
    out.update({
        "keyConfidence": p.key_confidence,
        "tuningOffsetCents": p.tuning_offset_cents,
        "beatTimesMs": p.beat_times_ms or [],
        "onsetTimesMs": p.onset_times_ms or [],
        "waveformPeaks": p.waveform_peaks or [],
        "analysisDiagnostics": p.analysis_diagnostics or {},
        "chords": [_chord_out(c, p.transpose_semitones, flats) for c in p.chords],
        "words": [_word_out(w) for w in p.words],
    })
    return out


@app.patch("/api/projects/{project_id}")
def patch_project(project_id: str, payload: S.ProjectPatch,
                  user: User = Depends(get_current_user),
                  session: Session = Depends(get_session)) -> Dict[str, Any]:
    p = _owned_project(project_id, user, session)
    if payload.title is not None:
        p.title = payload.title.strip()[:300] or p.title
    if payload.transpose_semitones is not None:
        # Transposition changes only how chords are *named*. The audio and
        # the stored chord data are untouched, which is what lets the user
        # transpose back and forth without any loss.
        p.transpose_semitones = payload.transpose_semitones
    session.commit()
    return get_project(project_id, user, session)


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str,
                   user: User = Depends(get_current_user),
                   session: Session = Depends(get_session)) -> Dict[str, Any]:
    p = _owned_project(project_id, user, session)
    storage.delete(p.storage_key)
    session.delete(p)
    session.commit()
    return {"ok": True}


@app.post("/api/projects/{project_id}/analyze")
def reanalyze(project_id: str, payload: S.AnalyzeIn,
              user: User = Depends(get_current_user),
              session: Session = Depends(get_session),
              _: None = Depends(rate_limit)) -> Dict[str, Any]:
    p = _owned_project(project_id, user, session)
    if p.status == ProjectStatus.ANALYZING.value:
        raise HTTPException(409, "This project is already being analysed.")

    job = AnalysisJob(project_id=p.id, stage="upload", progress=0.02,
                      status=JobStatus.QUEUED.value, stages=STAGES,
                      want_transcription=bool(payload.transcribe),
                      message="Queued for analysis")
    session.add(job)
    p.status = ProjectStatus.UPLOADED.value
    session.commit()
    get_queue().submit(job.id, p.id)
    return {"jobId": job.id}


# --- audio streaming -----------------------------------------------------------

@app.get("/api/projects/{project_id}/audio")
def stream_audio(project_id: str, request: Request,
                 user: User = Depends(get_current_user),
                 session: Session = Depends(get_session)):
    """Serve the audio with HTTP range support.

    Range support is not optional here: without it the browser cannot
    seek, so dragging the playhead would restart the download from zero
    every time.
    """
    p = _owned_project(project_id, user, session)
    path = storage.path_for(p.storage_key)
    if not os.path.isfile(path):
        raise HTTPException(404, "The audio file is missing from storage.")

    size = os.path.getsize(path)
    media_type = p.content_type or mimetypes.guess_type(p.file_name)[0] or "audio/mpeg"

    range_header = request.headers.get("range")
    start, end = 0, size - 1
    partial = False
    if range_header and range_header.startswith("bytes="):
        spec = range_header[6:].split(",")[0].strip()
        s, _, e = spec.partition("-")
        try:
            if s:
                start = int(s)
                end = int(e) if e else size - 1
            elif e:
                start = max(0, size - int(e))
            partial = True
        except ValueError:
            partial = False
        start = max(0, min(start, size - 1))
        end = max(start, min(end, size - 1))

    length = end - start + 1

    def body():
        with open(path, "rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Cache-Control": "private, max-age=3600",
    }
    if partial:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"

    return StreamingResponse(
        body(), status_code=206 if partial else 200,
        media_type=media_type, headers=headers,
    )


# --- progress (SSE) --------------------------------------------------------------

@app.get("/api/projects/{project_id}/events")
async def project_events(project_id: str,
                         user: User = Depends(get_current_user),
                         session: Session = Depends(get_session)):
    """Server-sent events carrying analysis progress.

    The worker writes progress into the job row; this reads it back. A
    short poll is used rather than an in-memory channel because the
    worker lives in another process, and because progress then survives
    the user reloading the page mid-analysis.
    """
    p = _owned_project(project_id, user, session)
    project_pk = p.id

    async def stream():
        from .db import SessionLocal
        last_payload: Optional[str] = None
        idle = 0.0
        yield f"event: stages\ndata: {json.dumps(STAGES)}\n\n"

        while idle < 1800:
            s = SessionLocal()
            try:
                job = s.scalars(
                    select(AnalysisJob)
                    .where(AnalysisJob.project_id == project_pk)
                    .order_by(AnalysisJob.created_at.desc()).limit(1)
                ).first()
                if job is None:
                    break
                payload = json.dumps({
                    "jobId": job.id, "stage": job.stage,
                    "progress": round(job.progress, 4), "status": job.status,
                    "message": job.message, "errorMessage": job.error_message,
                })
                if payload != last_payload:
                    last_payload = payload
                    yield f"event: progress\ndata: {payload}\n\n"
                    idle = 0.0
                if job.status in (JobStatus.SUCCEEDED.value, JobStatus.FAILED.value,
                                  JobStatus.CANCELLED.value):
                    yield f"event: done\ndata: {payload}\n\n"
                    return
            finally:
                s.close()

            await asyncio.sleep(0.4)
            idle += 0.4
            # A comment line keeps proxies from closing an idle stream.
            yield ": keepalive\n\n"

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


@app.get("/api/projects/{project_id}/job")
def latest_job(project_id: str,
               user: User = Depends(get_current_user),
               session: Session = Depends(get_session)) -> Dict[str, Any]:
    """Polling fallback for clients that cannot hold an SSE connection."""
    _owned_project(project_id, user, session)
    job = session.scalars(
        select(AnalysisJob).where(AnalysisJob.project_id == project_id)
        .order_by(AnalysisJob.created_at.desc()).limit(1)
    ).first()
    if job is None:
        raise HTTPException(404, "No analysis job for this project.")
    return {
        "id": job.id, "projectId": job.project_id, "stage": job.stage,
        "progress": round(job.progress, 4), "status": job.status,
        "message": job.message, "errorMessage": job.error_message,
        "stages": job.stages or STAGES,
    }


# --- the synchronised song view -----------------------------------------------

@app.get("/api/projects/{project_id}/song")
def song_view(project_id: str,
              user: User = Depends(get_current_user),
              session: Session = Depends(get_session)) -> Dict[str, Any]:
    """Chords placed above words, ready to render."""
    p = _owned_project(project_id, user, session)
    flats = _notation_flats(session, user.id)
    chords = [_chord_out(c, p.transpose_semitones, flats) for c in p.chords]
    words = [_word_out(w) for w in p.words]
    lines = synchronize(words, chords)
    return {
        "projectId": p.id, "title": p.title,
        "key": (f"{p.key_tonic} {p.key_mode}" if p.key_tonic else None),
        "keyConfidence": p.key_confidence,
        "tempoBpm": p.tempo_bpm,
        "transposeSemitones": p.transpose_semitones,
        "durationMs": p.duration_ms,
        "lines": [l.to_dict() for l in lines],
    }


# --- chord editing ---------------------------------------------------------------

@app.post("/api/projects/{project_id}/chords", status_code=201)
def add_chord(project_id: str, payload: S.ChordCreate,
              user: User = Depends(get_current_user),
              session: Session = Depends(get_session)) -> Dict[str, Any]:
    p = _owned_project(project_id, user, session)
    try:
        # The user types in the transposed key they are looking at, so
        # undo the transposition before storing.
        stored_name = transpose_chord(payload.chord_name, -p.transpose_semitones)
        root, quality, bass = parse_chord(stored_name)
    except ValueError as exc:
        raise HTTPException(400, f"Not a chord symbol this app understands: {exc}") from exc

    if payload.end_time_ms <= payload.start_time_ms:
        raise HTTPException(400, "The chord must end after it starts.")

    c = ChordEvent(
        project_id=p.id, chord_name=build_chord_name(root, quality, bass),
        root=pc_name(root), quality=quality,
        bass_note=pc_name(bass) if bass is not None else None,
        start_time_ms=payload.start_time_ms, end_time_ms=payload.end_time_ms,
        confidence=1.0, alternatives=[], is_edited=True,
    )
    session.add(c)
    session.commit()
    return _chord_out(c, p.transpose_semitones, _notation_flats(session, user.id))


@app.patch("/api/projects/{project_id}/chords/{chord_id}")
def patch_chord(project_id: str, chord_id: str, payload: S.ChordPatch,
                user: User = Depends(get_current_user),
                session: Session = Depends(get_session)) -> Dict[str, Any]:
    p = _owned_project(project_id, user, session)
    c = session.get(ChordEvent, chord_id)
    if c is None or c.project_id != p.id:
        raise HTTPException(404, "Chord not found.")

    if payload.chord_name is not None:
        try:
            stored_name = transpose_chord(payload.chord_name, -p.transpose_semitones)
            root, quality, bass = parse_chord(stored_name)
        except ValueError as exc:
            raise HTTPException(400, f"Not a chord symbol this app understands: {exc}") from exc
        c.chord_name = build_chord_name(root, quality, bass)
        c.root = pc_name(root)
        c.quality = quality
        c.bass_note = pc_name(bass) if bass is not None else None

    if payload.start_time_ms is not None:
        c.start_time_ms = max(0, payload.start_time_ms)
    if payload.end_time_ms is not None:
        c.end_time_ms = payload.end_time_ms
    if c.end_time_ms <= c.start_time_ms:
        raise HTTPException(400, "The chord must end after it starts.")

    c.is_edited = True
    session.commit()
    return _chord_out(c, p.transpose_semitones, _notation_flats(session, user.id))


@app.delete("/api/projects/{project_id}/chords/{chord_id}")
def delete_chord(project_id: str, chord_id: str,
                 user: User = Depends(get_current_user),
                 session: Session = Depends(get_session)) -> Dict[str, Any]:
    p = _owned_project(project_id, user, session)
    c = session.get(ChordEvent, chord_id)
    if c is None or c.project_id != p.id:
        raise HTTPException(404, "Chord not found.")
    session.delete(c)
    session.commit()
    return {"ok": True}


# --- lyric editing ----------------------------------------------------------------

@app.post("/api/projects/{project_id}/words", status_code=201)
def add_word(project_id: str, payload: S.WordCreate,
             user: User = Depends(get_current_user),
             session: Session = Depends(get_session)) -> Dict[str, Any]:
    p = _owned_project(project_id, user, session)
    if payload.end_time_ms <= payload.start_time_ms:
        raise HTTPException(400, "The word must end after it starts.")
    idx = session.scalar(
        select(func.count()).select_from(TranscriptWord)
        .where(TranscriptWord.project_id == p.id,
               TranscriptWord.line_id == payload.line_id)) or 0
    w = TranscriptWord(
        project_id=p.id, text=payload.text[:200],
        start_time_ms=payload.start_time_ms, end_time_ms=payload.end_time_ms,
        line_id=payload.line_id, word_index=idx,
        timing_source=TimingSource.USER.value, is_edited=True,
    )
    session.add(w)
    session.commit()
    return _word_out(w)


@app.patch("/api/projects/{project_id}/words/{word_id}")
def patch_word(project_id: str, word_id: str, payload: S.WordPatch,
               user: User = Depends(get_current_user),
               session: Session = Depends(get_session)) -> Dict[str, Any]:
    p = _owned_project(project_id, user, session)
    w = session.get(TranscriptWord, word_id)
    if w is None or w.project_id != p.id:
        raise HTTPException(404, "Word not found.")
    if payload.text is not None:
        w.text = payload.text[:200]
    if payload.start_time_ms is not None:
        w.start_time_ms = max(0, payload.start_time_ms)
        w.timing_source = TimingSource.USER.value
    if payload.end_time_ms is not None:
        w.end_time_ms = payload.end_time_ms
        w.timing_source = TimingSource.USER.value
    if payload.line_id is not None:
        w.line_id = payload.line_id
    if w.end_time_ms <= w.start_time_ms:
        raise HTTPException(400, "The word must end after it starts.")
    w.is_edited = True
    session.commit()
    return _word_out(w)


@app.post("/api/projects/{project_id}/words/{word_id}/split")
def split_word(project_id: str, word_id: str, payload: S.WordSplit,
               user: User = Depends(get_current_user),
               session: Session = Depends(get_session)) -> List[Dict[str, Any]]:
    p = _owned_project(project_id, user, session)
    w = session.get(TranscriptWord, word_id)
    if w is None or w.project_id != p.id:
        raise HTTPException(404, "Word not found.")

    ratio = min(max(payload.at_ratio, 0.05), 0.95)
    span = w.end_time_ms - w.start_time_ms
    cut = w.start_time_ms + int(round(span * ratio))

    text = w.text
    char_cut = max(1, min(len(text) - 1, int(round(len(text) * ratio)))) if len(text) > 1 else 1
    first = payload.first_text if payload.first_text is not None else text[:char_cut]
    second = payload.second_text if payload.second_text is not None else text[char_cut:]

    new = TranscriptWord(
        project_id=p.id, text=second[:200], start_time_ms=cut,
        end_time_ms=w.end_time_ms, line_id=w.line_id,
        word_index=w.word_index + 1, timing_source=TimingSource.USER.value,
        is_edited=True, language=w.language,
    )
    w.text = first[:200]
    w.end_time_ms = cut
    w.is_edited = True
    w.timing_source = TimingSource.USER.value

    for other in session.scalars(
        select(TranscriptWord).where(
            TranscriptWord.project_id == p.id,
            TranscriptWord.line_id == w.line_id,
            TranscriptWord.word_index > w.word_index)).all():
        other.word_index += 1

    session.add(new)
    session.commit()
    return [_word_out(w), _word_out(new)]


@app.delete("/api/projects/{project_id}/words/{word_id}")
def delete_word(project_id: str, word_id: str,
                user: User = Depends(get_current_user),
                session: Session = Depends(get_session)) -> Dict[str, Any]:
    p = _owned_project(project_id, user, session)
    w = session.get(TranscriptWord, word_id)
    if w is None or w.project_id != p.id:
        raise HTTPException(404, "Word not found.")
    session.delete(w)
    session.commit()
    return {"ok": True}


# --- export --------------------------------------------------------------------------

EXPORT_TYPES = {
    "pdf": ("application/pdf", "pdf"),
    "txt": ("text/plain; charset=utf-8", "txt"),
    "md": ("text/markdown; charset=utf-8", "md"),
    "json": ("application/json; charset=utf-8", "json"),
    "chordpro": ("text/plain; charset=utf-8", "pro"),
    "html": ("text/html; charset=utf-8", "html"),
}


@app.get("/api/projects/{project_id}/export/{fmt}")
def export_project(project_id: str, fmt: str, font_size: float = 11.0,
                   user: User = Depends(get_current_user),
                   session: Session = Depends(get_session)):
    if fmt not in EXPORT_TYPES:
        raise HTTPException(400, f"Unknown export format. Available: "
                                 f"{', '.join(sorted(EXPORT_TYPES))}")
    font_size = max(8.0, min(18.0, float(font_size)))
    p = _owned_project(project_id, user, session)
    view = song_view(project_id, user, session)

    doc = exporters.build_document(
        title=p.title, lines=view["lines"], key_name=view["key"],
        tempo_bpm=p.tempo_bpm, transpose_semitones=p.transpose_semitones,
    )
    doc.generated_note = "Generated by ChordVision"

    media_type, ext = EXPORT_TYPES[fmt]
    safe = "".join(ch for ch in p.title if ch.isalnum() or ch in " -_") .strip() or "song"
    from urllib.parse import quote
    disposition = (f'attachment; filename="song.{ext}"; '
                   f"filename*=UTF-8''{quote(safe + '.' + ext, safe='')}")

    if fmt == "pdf":
        try:
            data = exporters.to_pdf(doc, font_size=font_size)
        except Exception as exc:
            raise HTTPException(500, "יצירת PDF נכשלה. נסה פורמט אחר או נסה שוב.") from exc
        return Response(data, media_type=media_type,
                        headers={"Content-Disposition": disposition})

    if fmt == "json":
        body = exporters.to_json({
            "project": _project_summary(p, session),
            "key": view["key"], "tempoBpm": p.tempo_bpm,
            "transposeSemitones": p.transpose_semitones,
            "chords": view and [c for line in view["lines"] for c in line["chords"]],
            "lines": view["lines"],
            "analysis": p.analysis_diagnostics or {},
        })
    elif fmt == "txt":
        body = exporters.to_text(doc)
    elif fmt == "md":
        body = exporters.to_markdown(doc)
    elif fmt == "chordpro":
        body = exporters.to_chordpro(doc)
    else:
        body = exporters.to_html(doc)

    return Response(body, media_type=media_type,
                    headers={"Content-Disposition": disposition})


# --- static frontend ------------------------------------------------------------------

_FRONTEND_DIST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "frontend", "dist")

if os.path.isdir(_FRONTEND_DIST):
    # Hashed build assets are served directly and may be cached hard,
    # because their filenames change whenever their contents do.
    _ASSETS = os.path.join(_FRONTEND_DIST, "assets")
    if os.path.isdir(_ASSETS):
        app.mount("/assets", StaticFiles(directory=_ASSETS), name="assets")

    from fastapi.responses import FileResponse

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        """Serve the single-page app for every non-API path.

        The routing lives in the browser, so /song/<id> is a real address
        a person can bookmark, reload or share - but the server has no
        such route. Without this fallback a reload on a song page returns
        a JSON 404 instead of the app, which is the kind of break nobody
        notices until someone shares a link.
        """
        if full_path.startswith("api/"):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found.")

        candidate = os.path.normpath(os.path.join(_FRONTEND_DIST, full_path))
        if (full_path and candidate.startswith(_FRONTEND_DIST + os.sep)
                and os.path.isfile(candidate)):
            return FileResponse(candidate)

        return FileResponse(
            os.path.join(_FRONTEND_DIST, "index.html"),
            headers={"Cache-Control": "no-cache"},
        )
