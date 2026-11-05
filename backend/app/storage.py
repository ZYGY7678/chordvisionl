"""Where uploaded audio lives, and how it is validated on the way in.

Two rules from the specification are enforced here:

  * a file is never accepted on the strength of its extension - it has to
    decode as audio;
  * stored files are private, named by an unguessable key, and are served
    only through an endpoint that checks ownership.
"""
from __future__ import annotations

import os
import secrets
import shutil
from urllib.parse import quote
import httpx
from dataclasses import dataclass
from typing import BinaryIO, Optional, Tuple

from .config import get_settings

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga",
                      ".opus", ".webm", ".wma", ".aiff", ".aif"}

ALLOWED_CONTENT_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mp4", "audio/x-m4a", "audio/aac", "audio/flac", "audio/x-flac",
    "audio/ogg", "application/ogg", "audio/opus", "audio/webm", "video/webm",
    "audio/aiff", "audio/x-aiff", "application/octet-stream",
}


class UploadRejected(ValueError):
    """The upload is not acceptable. The message is safe to show the user."""


@dataclass
class StoredFile:
    storage_key: str
    path: str
    size_bytes: int
    duration_ms: int
    content_type: Optional[str]
    codec: str
    container: str
    sample_rate: int
    channels: int


def remote_storage_ready() -> bool:
    settings = get_settings()
    return bool(settings.supabase_url.strip() and settings.supabase_service_role_key.strip())


def _remote_headers(content_type: Optional[str] = None) -> dict:
    settings = get_settings()
    key = settings.supabase_service_role_key
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _remote_object_url(storage_key: str) -> str:
    settings = get_settings()
    base = settings.supabase_url.rstrip("/")
    bucket = quote(settings.supabase_storage_bucket, safe="")
    obj = quote(storage_key, safe="")
    return f"{base}/storage/v1/object/{bucket}/{obj}"


def _upload_remote(local_path: str, storage_key: str, content_type: Optional[str]) -> None:
    if not remote_storage_ready():
        return
    headers = _remote_headers(content_type or "application/octet-stream")
    headers["x-upsert"] = "false"
    try:
        with open(local_path, "rb") as fh, httpx.Client(timeout=180.0) as client:
            response = client.post(_remote_object_url(storage_key), headers=headers, content=fh)
            response.raise_for_status()
    except Exception as exc:
        raise UploadRejected("העלאת השיר לאחסון הקבוע נכשלה. נסה שוב.") from exc


def _restore_remote(storage_key: str, destination: str) -> None:
    if not remote_storage_ready():
        raise UploadRejected("קובץ השמע אינו זמין באחסון המקומי.")
    temporary = destination + ".download"
    try:
        with httpx.stream("GET", _remote_object_url(storage_key), headers=_remote_headers(), timeout=120.0) as response:
            response.raise_for_status()
            with open(temporary, "wb") as out:
                for chunk in response.iter_bytes(1024 * 1024):
                    if chunk:
                        out.write(chunk)
        os.replace(temporary, destination)
    except Exception as exc:
        _remove_quietly(temporary)
        raise UploadRejected("לא הצלחנו לשחזר את קובץ השיר מהאחסון הקבוע.") from exc


def _root() -> str:
    path = os.path.abspath(get_settings().storage_dir)
    os.makedirs(path, exist_ok=True)
    return path


def path_for(storage_key: str) -> str:
    """Resolve a storage key to a path, refusing anything that escapes the root."""
    root = _root()
    candidate = os.path.abspath(os.path.join(root, storage_key))
    if not candidate.startswith(root + os.sep):
        raise UploadRejected("Invalid storage key.")
    if not os.path.isfile(candidate) and remote_storage_ready():
        _restore_remote(storage_key, candidate)
    return candidate


def save_upload(fileobj: BinaryIO, original_name: str,
                content_type: Optional[str] = None) -> StoredFile:
    """Stream an upload to disk, then prove it is really audio.

    The size limit is enforced *while streaming*, not after, so an
    oversized upload cannot fill the disk before being rejected.
    """
    settings = get_settings()

    _, ext = os.path.splitext(original_name.lower())
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise UploadRejected(
            f"{ext} is not a supported audio format. Supported: "
            + ", ".join(sorted(e.lstrip('.') for e in ALLOWED_EXTENSIONS))
        )
    if content_type and content_type.split(";")[0].strip() not in ALLOWED_CONTENT_TYPES:
        raise UploadRejected(f"Unsupported content type: {content_type}")

    storage_key = f"{secrets.token_urlsafe(24)}{ext or '.bin'}"
    dest = path_for(storage_key)

    size = 0
    limit = settings.max_upload_bytes
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = fileobj.read(1024 * 256)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise UploadRejected(
                        f"The file is larger than the {limit // (1024*1024)} MB limit."
                    )
                out.write(chunk)
    except UploadRejected:
        _remove_quietly(dest)
        raise
    except OSError as exc:
        _remove_quietly(dest)
        raise UploadRejected(f"The file could not be stored: {exc}") from exc

    if size == 0:
        _remove_quietly(dest)
        raise UploadRejected("The file is empty.")

    # The real check: can this actually be decoded as audio?
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "engine"))
    from chordvision_engine.audio import probe, AudioDecodeError

    try:
        info = probe(dest)
    except AudioDecodeError as exc:
        _remove_quietly(dest)
        raise UploadRejected(
            "This file could not be decoded as audio. A correct extension is "
            f"not enough - the contents have to be real audio. ({exc})"
        ) from exc

    if info.duration_s <= 0.2:
        _remove_quietly(dest)
        raise UploadRejected("The audio is too short to analyse.")

    if info.duration_s > settings.max_audio_seconds:
        _remove_quietly(dest)
        raise UploadRejected(
            f"The audio is {info.duration_s / 60:.1f} minutes long; the limit is "
            f"{settings.max_audio_seconds / 60:.0f} minutes."
        )

    # Keep a durable private copy so a fresh Render instance can restore it.
    _upload_remote(dest, storage_key, content_type)

    return StoredFile(
        storage_key=storage_key,
        path=dest,
        size_bytes=size,
        duration_ms=int(round(info.duration_s * 1000)),
        content_type=content_type,
        codec=info.codec_name,
        container=info.format_name,
        sample_rate=info.sample_rate,
        channels=info.channels,
    )


def delete(storage_key: str) -> bool:
    local_removed = False
    try:
        path = path_for(storage_key)
        local_removed = _remove_quietly(path)
    except UploadRejected:
        pass
    if remote_storage_ready():
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.delete(_remote_object_url(storage_key), headers=_remote_headers())
                if response.status_code not in (200, 204, 404):
                    response.raise_for_status()
        except Exception:
            return False
    return local_removed or remote_storage_ready()


def _remove_quietly(path: str) -> bool:
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def compute_waveform_peaks(path: str, buckets: int = 1200) -> list:
    """Min/max pairs per bucket, for drawing the waveform.

    Peaks rather than averages: an averaged waveform of a loud, dense mix
    flattens into a featureless block, while peaks keep the shape of the
    arrangement visible, which is what makes the waveform useful for
    navigating.
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "engine"))
    import numpy as np
    from chordvision_engine.audio import decode

    samples, _sr = decode(path, target_sr=8000, mono=True)
    if samples.size == 0:
        return []

    buckets = max(1, min(buckets, samples.size))
    edges = np.linspace(0, samples.size, buckets + 1).astype(int)

    peaks = []
    for i in range(buckets):
        chunk = samples[edges[i]:edges[i + 1]]
        if chunk.size == 0:
            peaks.append([0.0, 0.0])
        else:
            peaks.append([round(float(chunk.min()), 4), round(float(chunk.max()), 4)])
    return peaks
