"""The analysis job queue and worker.

Why a separate worker at all
----------------------------
Analysing a song is seconds of sustained CPU.  Doing that inside a
request handler blocks the event loop, so every other user's requests
stall behind it.  The specification asks for a separate worker and a job
queue, and this provides both.

Where the work actually runs
----------------------------
Chord analysis runs in a **separate process**, through a
``ProcessPoolExecutor``.  A thread would not be enough: the Viterbi
decode is a Python loop, so it holds the interpreter lock and would stall
the server even though the numpy parts release it.

Transcription is network-bound, not CPU-bound, so it runs in the
coordinating thread with async HTTP.

Progress is written to the ``analysis_jobs`` row as it happens, and the
SSE endpoint reads it from there.  Routing progress through the database
rather than through an in-memory channel is what lets the worker live in
a different process, and is also what makes progress survive a page
reload.

Scope, stated plainly
---------------------
This queue is in-process: the pool belongs to the API server.  It handles
concurrent uploads correctly on one machine, which is what the project
needs, but it is not a distributed queue.  Running more than one API
instance means jobs are only visible to the instance that accepted them.
Moving to Celery or RQ with Redis means replacing ``submit`` and the
polling in the SSE route; everything else, including the stage model and
the progress rows, stays as it is.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Callable

from .config import get_settings
from .db import (
    SessionLocal, Project, AnalysisJob, ChordEvent, TranscriptWord,
    JobStatus, ProjectStatus, TimingSource, utcnow,
)

ENGINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "engine",
)
if ENGINE_PATH not in sys.path:
    sys.path.insert(0, ENGINE_PATH)


# The stages the UI shows. Each declares whether its progress can be
# measured or only observed starting and finishing: the specification
# forbids showing a fake percentage, so stages that cannot be measured
# report no number at all rather than a made-up one.
STAGES: List[Dict[str, Any]] = [
    {"id": "upload",            "label": "העלאת השיר",             "measurable": True},
    {"id": "prepare_audio",     "label": "הכנת קובץ השמע",         "measurable": True},
    {"id": "musical_analysis",  "label": "ניתוח מוזיקלי",          "measurable": True},
    {"id": "chord_detection",   "label": "זיהוי אקורדים",          "measurable": True},
    {"id": "transcription",     "label": "תמלול המילים",           "measurable": False},
    {"id": "word_alignment",    "label": "תזמון המילים",           "measurable": True},
    {"id": "synchronization",   "label": "סנכרון האקורדים והמילים", "measurable": True},
    {"id": "finalize",          "label": "הכנת תצוגת השיר",         "measurable": True},
]

STAGE_WEIGHT = {
    "upload": 0.02, "prepare_audio": 0.06, "musical_analysis": 0.26,
    "chord_detection": 0.22, "transcription": 0.30, "word_alignment": 0.08,
    "synchronization": 0.04, "finalize": 0.02,
}


# --- the function that runs in the worker process ---------------------------

def _analyze_in_subprocess(audio_path: str, quality_set: str,
                           progress_path: Optional[str],
                           max_audio_seconds: Optional[float] = None) -> Dict[str, Any]:
    """Run the chord engine. Executed in a separate process.

    Progress is written to a small text file rather than returned, since
    the parent cannot see inside a running pool task.  A file is used
    instead of a database handle because opening a second writer to the
    database from a short-lived worker is more fragile than writing one
    line to a file.
    """
    sys.path.insert(0, ENGINE_PATH)
    from chordvision_engine.engine_adapter import ChordEngine, EngineConfig

    def report(stage: str, frac: float, message: str) -> None:
        if not progress_path:
            return
        try:
            with open(progress_path, "w", encoding="utf-8") as fh:
                fh.write(f"{stage}\t{frac:.4f}\t{message}")
        except OSError:
            pass

    config = EngineConfig(quality_set=quality_set, max_duration_s=max_audio_seconds)
    result = ChordEngine(config).analyze_file(audio_path, progress=report)
    return result.to_dict()


# --- job coordination --------------------------------------------------------

@dataclass
class JobHandle:
    job_id: str
    project_id: str


class JobQueue:
    """Accepts analysis jobs and runs them off the request path."""

    def __init__(self, max_workers: Optional[int] = None):
        settings = get_settings()
        n = max_workers or settings.analysis_workers
        self._cpu_pool = ProcessPoolExecutor(max_workers=max(1, n))
        self._coordinator = ThreadPoolExecutor(
            max_workers=max(2, n * 2), thread_name_prefix="chordvision-job")
        self._closed = False

    def submit(self, job_id: str, project_id: str) -> JobHandle:
        if self._closed:
            raise RuntimeError("the job queue is shutting down")
        self._coordinator.submit(self._run, job_id, project_id)
        return JobHandle(job_id=job_id, project_id=project_id)

    def shutdown(self) -> None:
        self._closed = True
        self._coordinator.shutdown(wait=False, cancel_futures=True)
        self._cpu_pool.shutdown(wait=False, cancel_futures=True)

    # -- the job itself ------------------------------------------------

    def _run(self, job_id: str, project_id: str) -> None:
        settings = get_settings()
        session = SessionLocal()
        progress_path = os.path.join(
            os.path.abspath(settings.storage_dir), f".progress-{job_id}")

        try:
            job = session.get(AnalysisJob, job_id)
            project = session.get(Project, project_id)
            if job is None or project is None:
                return

            job.status = JobStatus.RUNNING.value
            job.started_at = utcnow()
            job.stages = STAGES
            project.status = ProjectStatus.ANALYZING.value
            session.commit()

            from . import storage
            audio_path = storage.path_for(project.storage_key)

            self._set(session, job, "prepare_audio", 0.0, "Reading the audio file")

            # Waveform for the player. Cheap, and the UI needs it early.
            try:
                project.waveform_peaks = storage.compute_waveform_peaks(audio_path)
                session.commit()
            except Exception:
                project.waveform_peaks = None
                session.commit()

            # ---- chord analysis, in its own process --------------------
            future = self._cpu_pool.submit(
                _analyze_in_subprocess, audio_path, settings.quality_set, progress_path,
                settings.max_audio_seconds
            )
            while not future.done():
                time.sleep(0.25)
                stage, frac, message = _read_progress(progress_path)
                if stage:
                    self._set(session, job, stage, frac, message)
            analysis = future.result()

            self._store_analysis(session, project, analysis)
            detected_count = sum(1 for c in analysis["chords"] if c["chordName"] != "N")
            self._set(session, job, "chord_detection", 1.0,
                      f"{detected_count} chords detected")

            # ---- transcription, if the user asked for it ----------------
            if job.want_transcription:
                self._set(session, job, "transcription", 0.0,
                          "Sending audio to the transcription provider")
                try:
                    words = self._transcribe_and_align(session, project, audio_path)
                    self._set(session, job, "word_alignment", 1.0,
                              f"{len(words)} words timed")
                except Exception as exc:
                    # A failed transcription must not throw away a
                    # perfectly good chord analysis.
                    job.error_message = f"Transcription failed: {exc}"
                    self._set(session, job, "transcription", 1.0,
                              "Transcription failed; chords are still available")

            self._set(session, job, "synchronization", 1.0, "Chords and lyrics aligned")

            project.status = ProjectStatus.READY.value
            job.status = JobStatus.SUCCEEDED.value
            job.stage = "finalize"
            job.progress = 1.0
            job.message = "Done"
            job.finished_at = utcnow()
            session.commit()

        except Exception as exc:
            session.rollback()
            try:
                job = session.get(AnalysisJob, job_id)
                project = session.get(Project, project_id)
                if job:
                    job.status = JobStatus.FAILED.value
                    job.error_message = f"{type(exc).__name__}: {exc}"[:2000]
                    job.message = "Analysis failed"
                    job.finished_at = utcnow()
                if project:
                    project.status = ProjectStatus.FAILED.value
                session.commit()
            except Exception:
                traceback.print_exc()
        finally:
            try:
                os.remove(progress_path)
            except OSError:
                pass
            session.close()

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _set(session, job: AnalysisJob, stage: str, frac: float, message: str) -> None:
        """Record progress, converting a within-stage fraction to overall."""
        done = 0.0
        for s in STAGES:
            if s["id"] == stage:
                break
            done += STAGE_WEIGHT.get(s["id"], 0.0)
        overall = done + STAGE_WEIGHT.get(stage, 0.0) * max(0.0, min(frac, 1.0))

        job.stage = stage
        job.progress = round(min(overall, 1.0), 4)
        job.message = message[:400]
        session.commit()

    @staticmethod
    def _store_analysis(session, project: Project, analysis: Dict[str, Any]) -> None:
        session.query(ChordEvent).filter(ChordEvent.project_id == project.id).delete()

        for c in analysis["chords"]:
            if c["chordName"] == "N":
                continue
            session.add(ChordEvent(
                project_id=project.id,
                chord_name=c["chordName"], root=c["root"], quality=c["quality"],
                bass_note=c["bassNote"],
                start_time_ms=c["startTimeMs"], end_time_ms=c["endTimeMs"],
                confidence=c["confidence"], alternatives=c["alternatives"],
                is_edited=False, detection_version=c["detectionVersion"],
            ))

        key = analysis["key"]
        project.key_tonic = key["tonic"]
        project.key_mode = key["mode"]
        project.key_confidence = key["confidence"]
        project.tempo_bpm = analysis["tempoBpm"]
        project.tuning_offset_cents = analysis["tuningOffsetCents"]
        project.beat_times_ms = analysis["beatTimesMs"]
        project.onset_times_ms = analysis["onsetTimesMs"]
        project.duration_ms = analysis["durationMs"] or project.duration_ms
        project.analysis_diagnostics = analysis["diagnostics"]
        session.commit()

    def _transcribe_and_align(self, session, project: Project,
                              audio_path: str) -> List[TranscriptWord]:
        """Fetch a transcript, give every word a time, and store the result."""
        import asyncio
        import numpy as np

        from .crypto import decrypt_api_key
        from .db import UserSettings
        from .transcription import get_provider, TranscriptionError
        from .alignment import (
            align_words_in_span, speech_energy_envelope, tokenize,
        )
        from .sync import group_words_into_lines

        settings = get_settings()
        us = session.get(UserSettings, project.user_id)
        if us is None or not us.encrypted_api_key or not us.transcription_provider:
            raise TranscriptionError(
                "No transcription provider is configured. Add an API key in settings."
            )
        if not us.allow_audio_upload:
            raise TranscriptionError(
                "Transcription needs permission to send this audio to "
                f"{us.transcription_provider}. Enable it in settings first."
            )

        api_key = decrypt_api_key(us.encrypted_api_key)
        provider = get_provider(us.transcription_provider, api_key,
                                settings.transcription_timeout_s)

        result = asyncio.run(provider.transcribe(
            audio_path, language=us.preferred_language or None))

        words_out: List[Dict[str, Any]] = []
        # When the provider delimits phrases, those are the line breaks.
        # Re-deriving lines from pauses would throw away better
        # information than we can reconstruct: the provider heard the
        # speech, we only see an energy envelope. ``line_of_word`` is
        # filled in when segments are available and left empty otherwise.
        line_of_word: List[int] = []

        if result.has_word_timings:
            for w in result.words:
                if not w.text.strip():
                    continue
                words_out.append({
                    "text": w.text, "startTimeMs": w.start_ms, "endTimeMs": w.end_ms,
                    "confidence": w.confidence,
                    "timingSource": TimingSource.PROVIDER_WORD.value,
                })
            if result.segments:
                bounds = [(s.start_ms, s.end_ms) for s in result.segments]
                for w in words_out:
                    idx = 0
                    for i, (bs, be) in enumerate(bounds):
                        if bs <= w["startTimeMs"] < be:
                            idx = i
                            break
                        if w["startTimeMs"] >= be:
                            idx = i
                    line_of_word.append(idx)
        else:
            # Only phrase-level times: run forced alignment inside each
            # phrase, which is exactly the case the specification calls out.
            from chordvision_engine.audio import decode as decode_audio
            samples, sr = decode_audio(audio_path, target_sr=16000, mono=True)
            envelope, fps = speech_energy_envelope(samples, sr)

            for seg_index, seg in enumerate(result.segments):
                tokens = tokenize(seg.text)
                if not tokens:
                    continue
                aligned = align_words_in_span(
                    tokens, seg.start_ms, seg.end_ms, envelope, fps,
                    language=result.language,
                )
                for aw in aligned:
                    words_out.append({
                        "text": aw.text, "startTimeMs": aw.start_ms,
                        "endTimeMs": aw.end_ms, "confidence": seg.confidence,
                        "timingSource": aw.timing_source,
                    })
                    line_of_word.append(seg_index)

        session.query(TranscriptWord).filter(
            TranscriptWord.project_id == project.id).delete()

        if line_of_word and len(line_of_word) == len(words_out):
            # Provider segments decide the lines. Long segments are still
            # wrapped, so one run-on phrase does not become an unreadable
            # line twenty words wide.
            grouped: List[List[Dict[str, Any]]] = []
            current_seg, buf = None, []
            for w, seg_index in zip(words_out, line_of_word):
                if seg_index != current_seg:
                    if buf:
                        grouped.extend(group_words_into_lines(buf))
                    buf, current_seg = [], seg_index
                buf.append(w)
            if buf:
                grouped.extend(group_words_into_lines(buf))
            lines = grouped
        else:
            lines = group_words_into_lines(words_out)

        stored: List[TranscriptWord] = []
        for line_id, line in enumerate(lines):
            for idx, w in enumerate(line):
                row = TranscriptWord(
                    project_id=project.id, text=w["text"],
                    start_time_ms=int(w["startTimeMs"]),
                    end_time_ms=int(w["endTimeMs"]),
                    confidence=w.get("confidence"),
                    line_id=line_id, word_index=idx,
                    timing_source=w["timingSource"],
                    language=result.language,
                )
                session.add(row)
                stored.append(row)
        session.commit()
        return stored


def _read_progress(path: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            parts = fh.read().split("\t")
        if len(parts) == 3:
            return parts[0], float(parts[1]), parts[2]
    except (OSError, ValueError):
        pass
    return None, 0.0, ""


_QUEUE: Optional[JobQueue] = None


def recover_interrupted_jobs() -> int:
    """Fail every job that a previous process left unfinished.

    The queue lives inside the API process, so a restart - a deploy, a
    crash, an out-of-memory kill - silently abandons whatever was queued or
    running.  Without this the database keeps saying "analysing" forever
    and the user stares at a bar that will never move.  Marking those jobs
    failed makes the UI offer its normal retry button instead.

    Call it once at startup, before the queue accepts work.  Returns how
    many jobs were recovered.
    """
    settings = get_settings()
    recovered = 0
    session = SessionLocal()
    try:
        stuck = session.query(AnalysisJob).filter(
            AnalysisJob.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value])
        ).all()
        for job in stuck:
            job.status = JobStatus.FAILED.value
            job.message = "Interrupted by a server restart"
            job.error_message = (
                "The server restarted while this analysis was running. "
                "Run the analysis again.")
            job.finished_at = utcnow()
            project = session.get(Project, job.project_id)
            if project is not None and project.status in (
                    ProjectStatus.ANALYZING.value, ProjectStatus.UPLOADED.value):
                project.status = ProjectStatus.FAILED.value
            recovered += 1
        session.commit()
    finally:
        session.close()

    # Progress scratch files that belonged to those jobs.
    storage_dir = os.path.abspath(settings.storage_dir)
    try:
        for name in os.listdir(storage_dir):
            if name.startswith(".progress-"):
                try:
                    os.remove(os.path.join(storage_dir, name))
                except OSError:
                    pass
    except OSError:
        pass
    return recovered


def get_queue() -> JobQueue:
    global _QUEUE
    if _QUEUE is None:
        _QUEUE = JobQueue()
    return _QUEUE


def shutdown_queue() -> None:
    global _QUEUE
    if _QUEUE is not None:
        _QUEUE.shutdown()
        _QUEUE = None
