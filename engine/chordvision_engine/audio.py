"""Audio decoding and PCM preparation.

Decoding is delegated to ffmpeg, which is the only dependable way to accept
the full format list the project requires (MP3, WAV, M4A, AAC, FLAC, OGG,
WEBM) including the many container/codec combinations that appear in the
wild.  We never trust the file extension: the file is probed, and a file
that ffmpeg cannot decode into real audio is rejected.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np


class AudioDecodeError(RuntimeError):
    """Raised when a file cannot be decoded into usable audio."""


@dataclass
class AudioProbe:
    """What ffprobe tells us about the file, before we trust anything."""
    format_name: str
    codec_name: str
    duration_s: float
    sample_rate: int
    channels: int
    bit_rate: Optional[int]
    raw: Dict[str, Any]


def _ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise AudioDecodeError(
            "ffmpeg was not found on PATH. The server requires ffmpeg to decode audio."
        )
    return exe


def _ffprobe_bin() -> str:
    exe = shutil.which("ffprobe")
    if not exe:
        raise AudioDecodeError(
            "ffprobe was not found on PATH. The server requires ffprobe to inspect audio."
        )
    return exe


def probe(path: str) -> AudioProbe:
    """Inspect a media file. Raises AudioDecodeError if it holds no audio stream."""
    if not os.path.isfile(path):
        raise AudioDecodeError(f"File not found: {path}")

    cmd = [
        _ffprobe_bin(),
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-select_streams", "a:0",
        path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError("Probing the file timed out.") from exc

    if out.returncode != 0:
        raise AudioDecodeError(
            "The file could not be read as media: "
            + out.stderr.decode("utf-8", "replace").strip()[:400]
        )

    try:
        info = json.loads(out.stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        raise AudioDecodeError("ffprobe returned output that could not be parsed.") from exc

    streams = info.get("streams") or []
    if not streams:
        raise AudioDecodeError(
            "The file contains no audio stream. A real audio file is required; "
            "the extension alone is not enough."
        )

    stream = streams[0]
    fmt = info.get("format", {})

    duration = stream.get("duration") or fmt.get("duration")
    try:
        duration_s = float(duration) if duration is not None else 0.0
    except (TypeError, ValueError):
        duration_s = 0.0

    try:
        sr = int(stream.get("sample_rate") or 0)
    except (TypeError, ValueError):
        sr = 0

    bit_rate = stream.get("bit_rate") or fmt.get("bit_rate")
    try:
        bit_rate_i = int(bit_rate) if bit_rate is not None else None
    except (TypeError, ValueError):
        bit_rate_i = None

    return AudioProbe(
        format_name=str(fmt.get("format_name", "unknown")),
        codec_name=str(stream.get("codec_name", "unknown")),
        duration_s=duration_s,
        sample_rate=sr,
        channels=int(stream.get("channels") or 0),
        bit_rate=bit_rate_i,
        raw=info,
    )


def decode(
    path: str,
    target_sr: int = 11025,
    mono: bool = True,
    max_seconds: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """Decode any supported audio file to float32 PCM at ``target_sr``.

    Returns ``(samples, sample_rate)``. ``samples`` is 1-D when ``mono`` is
    True, otherwise shape ``(channels, n)``.

    ffmpeg performs the resampling with its own high quality polyphase
    resampler (soxr when available), which is better than anything we would
    write here and keeps the decode path to a single process.
    """
    info = probe(path)
    if info.channels <= 0:
        raise AudioDecodeError("The audio stream reports no channels.")

    channels = 1 if mono else min(info.channels, 2)

    cmd = [
        _ffmpeg_bin(),
        "-v", "error",
        "-nostdin",
    ]
    if max_seconds is not None:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += [
        "-i", path,
        "-map", "0:a:0",
        "-vn",
        "-ac", str(channels),
        "-ar", str(target_sr),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-",
    ]

    try:
        out = subprocess.run(cmd, capture_output=True, timeout=1800, check=False)
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError("Decoding the file timed out.") from exc

    if out.returncode != 0:
        raise AudioDecodeError(
            "ffmpeg could not decode the audio: "
            + out.stderr.decode("utf-8", "replace").strip()[:400]
        )

    buf = np.frombuffer(out.stdout, dtype="<f4")
    if buf.size == 0:
        raise AudioDecodeError("Decoding produced no audio samples.")

    if channels > 1:
        usable = (buf.size // channels) * channels
        samples = buf[:usable].reshape(-1, channels).T.copy()
    else:
        samples = buf.astype(np.float32, copy=True)

    # Guard against NaN / inf from damaged files.
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    return samples, target_sr


def stereo_mid_side(stereo: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split an (2, n) stereo signal into mid and side channels.

    The mid channel carries what is common to both speakers (usually lead
    vocal, bass and kick); the side channel carries what is panned.  The
    harmonic engine prefers the mid channel for the bass register and the
    full mix elsewhere, because the side channel is where most of the
    reverb and stereo-spread noise lives.
    """
    if stereo.ndim != 2 or stereo.shape[0] < 2:
        raise ValueError("stereo_mid_side expects an array shaped (2, n)")
    left = stereo[0]
    right = stereo[1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    return mid.astype(np.float32), side.astype(np.float32)


def peak_normalize(x: np.ndarray, target_peak: float = 0.98) -> np.ndarray:
    """Scale so the loudest sample sits at ``target_peak``. Silence is left alone."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 1e-9:
        return x
    return (x * (target_peak / peak)).astype(np.float32)


def rms_db(x: np.ndarray) -> float:
    """Full-signal RMS in dBFS, used to detect near-silent uploads."""
    if x.size == 0:
        return -np.inf
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    if rms <= 1e-12:
        return -np.inf
    return 20.0 * float(np.log10(rms))
