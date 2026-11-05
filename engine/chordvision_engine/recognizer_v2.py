"""
Chord recognition engine
========================
Pipeline:
  1. Load audio (mono, 11.025 kHz)
  2. Harmonic/percussive separation  -> drums stop polluting the chroma
  3. Tuning estimation               -> songs that are not tuned to A=440 still work
  4. High-resolution CQT chroma (36 bins/octave) for the full range + a separate "bass chroma"
  5. Beat tracking + beat-synchronous median chroma -> stable chord segments
  6. Key estimation (Krumhansl-Schmuckler) -> small prior for in-key chords
  7. Template matching with overtone-aware chord templates (+ bass-note bonus)
  8. Viterbi decoding -> smooth chord sequence, no flickering
Output: list of {start, end, chord} plus tempo / key / tuning.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Callable, Dict, Any
import numpy as np

ProgressFn = Callable[[str, float, str], None]

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# chord type -> (semitone intervals from root, suffix shown to the user, triad family)
CHORD_TYPES = {
    "maj":  ([0, 4, 7],      "",     "maj"),
    "min":  ([0, 3, 7],      "m",    "min"),
    "7":    ([0, 4, 7, 10],  "7",    "maj"),
    "maj7": ([0, 4, 7, 11],  "maj7", "maj"),
    "min7": ([0, 3, 7, 10],  "m7",   "min"),
    "dim":  ([0, 3, 6],      "dim",  "dim"),
    "sus4": ([0, 5, 7],      "sus4", None),
    "sus2": ([0, 2, 7],      "sus2", None),
    "aug":  ([0, 4, 8],      "aug",  None),
    "6":    ([0, 4, 7, 9],   "6",    "maj"),
    "min6": ([0, 3, 7, 9],   "m6",   "min"),
    "dim7": ([0, 3, 6, 9],   "dim7", "dim"),
    "m7b5": ([0, 3, 6, 10],  "m7b5", "dim"),
}

VOCABULARIES = {
    "simple":   ["maj", "min"],
    "standard": ["maj", "min", "7", "maj7", "min7"],
    "sevenths": ["maj", "min", "7", "maj7", "min7", "dim", "aug", "sus2", "sus4", "m7b5", "dim7"],
    "triads":   ["maj", "min"],
    "triads_ext": ["maj", "min", "dim", "aug", "sus2", "sus4"],
    "full":     list(CHORD_TYPES.keys()),
}

# Krumhansl-Kessler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class Segment:
    start: float
    end: float
    chord: str
    confidence: float = 0.0
    alternatives: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Result:
    chords: List[Segment]
    tempo: float
    key: str
    tuning_cents: float
    duration: float
    beats: List[float] = None
    onsets: List[float] = None

    def to_dict(self):
        d = asdict(self)
        for s in d["chords"]:
            s["start"] = round(s["start"], 3)
            s["end"] = round(s["end"], 3)
        d["tempo"] = round(float(d["tempo"]), 1)
        d["tuning_cents"] = round(float(d["tuning_cents"]), 1)
        d["duration"] = round(float(d["duration"]), 2)
        d["beats"] = [round(float(b), 3) for b in (d["beats"] or [])]
        d["onsets"] = [round(float(b), 3) for b in (d["onsets"] or [])]
        for segment in d["chords"]:
            segment["start"] = round(float(segment["start"]), 3)
            segment["end"] = round(float(segment["end"]), 3)
            segment["confidence"] = round(float(segment["confidence"]), 4)
        return d


# --------------------------------------------------------------------------- templates
def _build_templates(types: List[str], seventh_weight: float = 0.4, overtones: bool = True):
    """Overtone-aware templates: every chord tone also contributes its first harmonics."""
    # harmonic h of a note lands on these pitch classes (relative): 1:+0, 2:+0, 3:+7, 4:+0
    harmonics = [(0, 1.0), (0, 0.5), (7, 0.33), (0, 0.25)] if overtones else [(0, 1.0)]
    labels, templates, roots, families = [], [], [], []
    for root in range(12):
        for t in types:
            intervals, suffix, family = CHORD_TYPES[t]
            v = np.zeros(12)
            for i, iv in enumerate(intervals):
                # root a bit stronger, 7th a bit weaker (it is usually quieter in real music)
                w = 1.0 if i == 0 else (seventh_weight if i == 3 and len(intervals) == 4 and t not in ("6", "min6", "dim7") else 0.9)
                for off, hw in harmonics:
                    v[(root + iv + off) % 12] += w * hw
            v /= np.linalg.norm(v)
            labels.append(NOTES[root] + suffix)
            templates.append(v)
            roots.append(root)
            families.append(family)
    return labels, np.array(templates), np.array(roots), families


# --------------------------------------------------------------------------- key
def _estimate_key(chroma_mean: np.ndarray):
    best, best_score = (0, "major"), -2
    for tonic in range(12):
        for mode, prof in (("major", _MAJOR), ("minor", _MINOR)):
            sc = np.corrcoef(chroma_mean, np.roll(prof, tonic))[0, 1]
            if sc > best_score:
                best, best_score = (tonic, mode), sc
    return best


def _diatonic_set(tonic: int, mode: str):
    if mode == "major":
        degs = [(0, "maj"), (2, "min"), (4, "min"), (5, "maj"), (7, "maj"), (9, "min"), (11, "dim")]
    else:  # natural + harmonic minor
        degs = [(0, "min"), (2, "dim"), (3, "maj"), (5, "min"), (7, "min"), (7, "maj"), (8, "maj"), (10, "maj")]
    return {((tonic + d) % 12, fam) for d, fam in degs}


# --------------------------------------------------------------------------- viterbi
def _viterbi(log_emit: np.ndarray, log_trans: np.ndarray) -> np.ndarray:
    T, S = log_emit.shape
    delta = np.empty((T, S))
    psi = np.zeros((T, S), dtype=int)
    delta[0] = log_emit[0] - np.log(S)
    for t in range(1, T):
        scores = delta[t - 1][:, None] + log_trans
        psi[t] = scores.argmax(0)
        delta[t] = scores.max(0) + log_emit[t]
    path = np.empty(T, dtype=int)
    path[-1] = delta[-1].argmax()
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


# --------------------------------------------------------------------------- main
def _note_dictionary(n_notes=72, bpo=36, n_bins=216, decay=0.6, n_harm=10):
    """Each column = the CQT 'fingerprint' of one note with its overtones (for NNLS note transcription)."""
    D = np.zeros((n_bins, n_notes))
    per_semi = bpo / 12.0
    for n in range(n_notes):
        for h in range(1, n_harm + 1):
            pos = per_semi * n + bpo * np.log2(h)
            amp = decay ** (h - 1)
            for k in range(int(pos) - 2, int(pos) + 4):
                if 0 <= k < n_bins:
                    D[k, n] += amp * np.exp(-0.5 * ((k - pos) / 0.8) ** 2)
    return D


def _harmonic_chunked(
    y: np.ndarray,
    margin: float,
    sr: int,
    chunk_s: float = 8.0,
    pad_s: float = 1.5,
    progress: Optional[ProgressFn] = None,
):
    """Separate harmonic audio in short overlapping blocks.

    Librosa's HPSS builds several temporary spectrograms. Processing an
    entire recording at once can exceed the 512 MiB memory limit of a small
    Render instance, so only one short block is held by the separator at a
    time. The overlap keeps median-filter edge effects outside the copied
    portion.
    """
    import librosa

    n = int(y.size)
    chunk = max(1, int(chunk_s * sr))
    pad = max(0, int(pad_s * sr))
    out = np.empty(n, dtype=np.float32)
    starts = list(range(0, n, chunk))
    total = len(starts)

    for index, start in enumerate(starts, 1):
        core_end = min(n, start + chunk)
        a = max(0, start - pad)
        b = min(n, core_end + pad)
        block = y[a:b]
        frame_length = max(512, int(round(2048 * sr / 22050)))
        hpss_hop = max(128, frame_length // 4)
        harmonic = librosa.effects.harmonic(
            block, margin=margin, n_fft=frame_length, hop_length=hpss_hop
        )
        left = start - a
        right = left + (core_end - start)
        out[start:core_end] = harmonic[left:right]
        del block, harmonic

        if progress is not None:
            fraction = index / max(total, 1)
            progress(
                "musical_analysis",
                0.05 + 0.22 * fraction,
                f"Separating harmonic content: block {index}/{total}",
            )

    return out


def _cqt_chunked(
    y: np.ndarray,
    sr: int,
    hop: int,
    tuning: float,
    progress: Optional[ProgressFn] = None,
    chunk_s: float = 12.0,
    pad_s: float = 2.0,
) -> np.ndarray:
    """Compute a time-aligned CQT without materializing a full-song STFT.

    Blocks begin and end on the global hop grid. A two-second context protects
    the low C2 filters from block edges; only central frames are copied into
    the final compact float32 matrix.
    """
    import librosa

    n_frames = int(y.size // hop) + 1
    n_bins = 216
    result = np.empty((n_bins, n_frames), dtype=np.float32)
    core_frames = max(1, int(round(chunk_s * sr / hop)))
    pad_frames = max(1, int(round(pad_s * sr / hop)))
    starts = list(range(0, n_frames, core_frames))
    total = len(starts)

    f_min = librosa.note_to_hz("C2")
    for index, start_frame in enumerate(starts, 1):
        end_frame = min(n_frames, start_frame + core_frames)
        a_frame = max(0, start_frame - pad_frames)
        b_frame = min(n_frames, end_frame + pad_frames)
        a_sample = a_frame * hop
        b_sample = min(int(y.size), b_frame * hop)
        block = y[a_sample:b_sample]

        local = np.abs(librosa.cqt(
            block,
            sr=sr,
            hop_length=hop,
            fmin=f_min,
            n_bins=n_bins,
            bins_per_octave=36,
            tuning=tuning * 3.0,
        )).astype(np.float32, copy=False)

        local_start = start_frame - a_frame
        count = end_frame - start_frame
        local_end = local_start + count
        if local_end > local.shape[1]:
            raise RuntimeError(
                "Chunked CQT returned fewer frames than expected; "
                "refusing to publish an incomplete chord analysis."
            )
        result[:, start_frame:end_frame] = local[:, local_start:local_end]
        del block, local

        if progress is not None:
            fraction = index / max(total, 1)
            progress(
                "musical_analysis",
                0.30 + 0.34 * fraction,
                f"Calculating high-resolution CQT: block {index}/{total}",
            )

    return result


def _onset_strength_chunked(
    y: np.ndarray,
    sr: int,
    hop: int,
    progress: Optional[ProgressFn] = None,
    chunk_s: float = 18.0,
    pad_s: float = 1.0,
) -> np.ndarray:
    """Calculate a full-length onset envelope in small mel-spectrogram blocks."""
    import librosa

    n_frames = int(y.size // hop) + 1
    out = np.empty(n_frames, dtype=np.float32)
    core_frames = max(1, int(round(chunk_s * sr / hop)))
    pad_frames = max(1, int(round(pad_s * sr / hop)))
    starts = list(range(0, n_frames, core_frames))
    total = len(starts)

    for index, start_frame in enumerate(starts, 1):
        end_frame = min(n_frames, start_frame + core_frames)
        a_frame = max(0, start_frame - pad_frames)
        b_frame = min(n_frames, end_frame + pad_frames)
        block = y[a_frame * hop:min(int(y.size), b_frame * hop)]
        frame_length = max(512, int(round(2048 * sr / 22050)))
        local = librosa.onset.onset_strength(
            y=block, sr=sr, hop_length=hop, n_fft=frame_length
        )

        local_start = start_frame - a_frame
        count = end_frame - start_frame
        available = max(0, min(count, local.size - local_start))
        if available:
            out[start_frame:start_frame + available] = local[local_start:local_start + available]
        if available < count:
            out[start_frame + available:end_frame] = 0.0
        del block, local

        if progress is not None:
            fraction = index / max(total, 1)
            progress(
                "musical_analysis",
                0.66 + 0.25 * fraction,
                f"Building onset map: block {index}/{total}",
            )

    return out



def _estimate_beats_low_memory(
    onset_env: np.ndarray,
    sr: int,
    hop: int,
    min_bpm: float = 40.0,
    max_bpm: float = 220.0,
) -> tuple[float, np.ndarray]:
    """Estimate a global beat period and phase using a compact autocorrelation.

    librosa.beat.beat_track constructs a dense tempogram whose temporary
    allocations can exceed a 512 MiB container for long recordings. The
    autocorrelation below needs only O(N) memory and the search is restricted
    to musically useful beat periods. It returns frame indices on the same
    analysis grid used by the chord synchronizer.
    """
    from scipy.fft import next_fast_len, rfft, irfft
    from scipy.signal import find_peaks

    env = np.asarray(onset_env, dtype=np.float32).reshape(-1)
    if env.size < 8 or not np.any(env):
        period = max(1, int(round((sr / hop) * 0.5)))
        return 120.0, np.arange(0, env.size, period, dtype=np.int64)

    # Remove the DC component and weak background activity before measuring
    # periodicity. A small non-negative floor is useful for quiet recordings.
    floor = float(np.percentile(env, 35))
    signal = np.maximum(env - floor, 0.0).astype(np.float32, copy=False)
    signal -= np.mean(signal, dtype=np.float64).astype(np.float32)

    fps = float(sr) / float(hop)
    min_lag = max(2, int(np.floor(fps * 60.0 / max_bpm)))
    max_lag = min(env.size // 2, int(np.ceil(fps * 60.0 / min_bpm)))
    if max_lag <= min_lag:
        period = max(1, int(round(fps * 0.5)))
        return 120.0, np.arange(0, env.size, period, dtype=np.int64)

    fft_size = next_fast_len(2 * env.size - 1)
    spectrum = rfft(signal, n=fft_size)
    ac = irfft(spectrum * np.conjugate(spectrum), n=fft_size)[:max_lag + 1]
    # Correct for the decreasing overlap at longer lags.
    lags = np.arange(min_lag, max_lag + 1, dtype=np.int64)
    scores = ac[min_lag:max_lag + 1] / np.maximum(env.size - lags, 1)
    peaks, _ = find_peaks(scores)
    if peaks.size:
        peak_scores = scores[peaks]
        strongest = float(np.max(peak_scores))
        # Exact or near-exact periodic signals often produce equally strong
        # peaks at the beat and its half-time multiples. Prefer the shorter
        # period only when the scores are effectively tied; real music with
        # a pronounced half-time accent can still select the stronger peak.
        tied = peaks[peak_scores >= strongest * 0.995] if strongest > 0 else peaks
        best = int(np.min(tied)) + min_lag
    else:
        best = int(np.argmax(scores)) + min_lag

    tempo = 60.0 * fps / float(best)
    # Find the strongest phase in a folded onset envelope.
    phase_matrix = np.zeros(((env.size + best - 1) // best, best), dtype=np.float32)
    phase_matrix.reshape(-1)[:env.size] = signal
    phase = int(np.argmax(np.sum(phase_matrix, axis=0, dtype=np.float64)))
    beats = np.arange(phase, env.size, best, dtype=np.int64)
    return float(tempo), beats

def _frame_rms_chunked(
    y: np.ndarray,
    hop: int,
    frame_length: int = 2048,
    batch_frames: int = 512,
) -> np.ndarray:
    """Centered RMS with bounded temporary memory (no whole-song frame matrix)."""
    n_frames = int(y.size // hop) + 1
    half = frame_length // 2
    padded = np.pad(np.asarray(y, dtype=np.float32), (half, half), mode="constant")
    offsets = np.arange(frame_length, dtype=np.int64)
    out = np.empty(n_frames, dtype=np.float32)

    for first in range(0, n_frames, batch_frames):
        last = min(n_frames, first + batch_frames)
        centers = np.arange(first, last, dtype=np.int64) * hop
        frames = padded[centers[:, None] + offsets[None, :]]
        out[first:last] = np.sqrt(np.mean(frames * frames, axis=1, dtype=np.float32))
        del frames
    return out

def compute_features(
    path: Optional[str] = None,
    max_seconds: Optional[float] = 600,
    sr: int = 22050,
    mode: str = "nnls",          # "nnls" (best) or "cqt" (fast, simple chroma)
    harm_margin: float = 3.0,
    decay: float = 0.6,
    compress: float = 0.5,
    samples: Optional[np.ndarray] = None,
    progress: Optional[ProgressFn] = None,
):
    import librosa
    from scipy.optimize import nnls

    def emit(stage: str, fraction: float, message: str) -> None:
        if progress is not None:
            progress(stage, float(np.clip(fraction, 0.0, 1.0)), message)

    if samples is None:
        emit("prepare_audio", 0.05, "Decoding audio for the new chord engine")
        if not path:
            raise ValueError("An audio path or decoded samples are required")
        y, sr = librosa.load(path, sr=sr, mono=True, duration=max_seconds)
    else:
        y = np.asarray(samples, dtype=np.float32)
        if y.ndim != 1:
            raise ValueError("The chord engine requires mono audio")
        if max_seconds is not None:
            y = y[: int(max_seconds * sr)]
    if y.size < sr:  # shorter than 1 s
        raise ValueError("Audio is too short; at least one second is required")
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    duration = len(y) / sr
    # Keep about 43 analysis frames per second while decoding at 11.025 kHz.
    # Chord CQT bins only extend to B7 (~2.0 kHz), leaving ample Nyquist
    # headroom while halving the decoded waveform memory footprint.
    hop = max(128, int(round(sr / 43.06640625)))
    if samples is None:
        emit("prepare_audio", 1.0, "Audio decoded and prepared")

    # 1. harmonic part only (removes drums / transients)
    emit("musical_analysis", 0.05, "Separating harmonic content from percussion")
    y_h = _harmonic_chunked(y, harm_margin, sr, progress=progress)
    emit("musical_analysis", 0.28, "Harmonic audio isolated")

    # 2. tuning. A representative opening minute is enough for a global
    # tuning estimate and avoids constructing a full-song pitch track.
    tuning_probe = y_h[:min(y_h.size, 60 * sr)]
    tuning = float(librosa.estimate_tuning(y=tuning_probe, sr=sr, bins_per_octave=12))
    del tuning_probe
    emit("musical_analysis", 0.30, "Tuning estimate calculated")

    # 3. constant-Q spectrum in short, overlapping blocks. This avoids the
    # large temporary STFT/filter-bank allocations made by a whole-song CQT.
    C = _cqt_chunked(y_h, sr, hop, tuning, progress=progress)
    n_frames = C.shape[1]
    emit("musical_analysis", 0.65, "High-resolution CQT calculated")

    # 4. onset map in small blocks. librosa's default onset-strength path
    # builds a mel spectrogram; doing that for a five-minute track at once
    # creates a large temporary matrix on a 512 MiB instance.
    onset_env = _onset_strength_chunked(y, sr, hop, progress=progress)
    tempo, beats = _estimate_beats_low_memory(onset_env, sr, hop)
    if len(beats) < 4:  # beat tracker failed -> fixed 0.5 s grid
        beats = np.arange(0, n_frames, int(0.5 * sr / hop))
        tempo = 120.0
    elif tempo > 150:   # very fast: chords rarely change faster than every 2 beats
        beats = beats[::2]
    frames = np.unique(librosa.util.fix_frames(beats, x_min=0, x_max=n_frames))
    emit("musical_analysis", 1.0, "Beat grid and onset map calculated")

    # 5. beat-synchronous spectrum  ->  chroma
    Cfr = np.power(C, compress).astype(np.float32, copy=False)             # frame-level (for precise timing)
    spec = librosa.util.sync(C, frames, aggregate=np.mean)                  # (216, T)
    np.power(spec, compress, out=spec)
    spec = spec.astype(np.float32, copy=False)
    del C
    T = spec.shape[1]
    if mode == "nnls":
        # note transcription: spec ~= D @ notes, notes >= 0  (removes overtones explicitly)
        emit("chord_detection", 0.03, "Separating fundamental notes with NNLS")
        D = _note_dictionary(decay=decay)
        notes = np.zeros((D.shape[1], T), dtype=np.float32)
        report_every = max(1, T // 30)
        for t in range(T):
            notes[:, t] = nnls(D, spec[:, t], maxiter=400)[0]
            if (t + 1) % report_every == 0 or t + 1 == T:
                emit("chord_detection", 0.03 + 0.52 * ((t + 1) / max(T, 1)),
                     f"Fundamental-note separation: {t + 1}/{T} beat windows")
        chroma = np.zeros((12, T))
        bass = np.zeros((12, T))
        for n in range(60):                     # C2..B6 for harmony
            chroma[n % 12] += notes[n]
        for n in range(30):                     # C2..F#4 -> bass-ish, lower notes weigh more
            bass[n % 12] += notes[n] * (1.0 - n / 36.0)
    else:
        chroma = np.zeros((12, T)); bass = np.zeros((12, T))
        for k in range(216):
            pc = int(round(k / 3.0)) % 12
            chroma[pc] += spec[k]
            if k < 90: bass[pc] += spec[k]
    c_sync, b_sync = chroma, bass

    rms_frame_length = max(512, int(round(2048 * sr / 22050)))
    rms = _frame_rms_chunked(y_h, hop, frame_length=rms_frame_length)[None, :]
    r_sync = librosa.util.sync(rms, frames, aggregate=np.mean)[0]

    return dict(c_sync=c_sync, b_sync=b_sync, r_sync=r_sync, frames=frames, tempo=tempo,
                tuning=tuning, duration=duration, sr=sr, hop=hop, mode=mode, Cfr=Cfr,
                beat_times=librosa.frames_to_time(beats, sr=sr, hop_length=hop),
                onset_frames=librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop, backtrack=False))


def decode(feat, vocab="standard", beta=14.0, p_stay=0.80, key_bonus=0.06, bass_weight=0.22,
           min_segment=1.0, seventh_weight=0.6, ext_penalty=0.04, slash=False, refine=True, snap_onsets=False,
           progress: Optional[ProgressFn] = None) -> Result:
    import librosa

    def emit(fraction: float, message: str) -> None:
        if progress is not None:
            progress("chord_detection", float(np.clip(fraction, 0.0, 1.0)), message)

    c_sync, b_sync, r_sync = feat["c_sync"], feat["b_sync"], feat["r_sync"]
    frames, tempo, tuning = feat["frames"], feat["tempo"], feat["tuning"]
    duration, sr, hop = feat["duration"], feat["sr"], feat["hop"]
    T = c_sync.shape[1]

    # normalise every beat
    norm = np.linalg.norm(c_sync, axis=0, keepdims=True)
    c_norm = c_sync / np.maximum(norm, 1e-9)
    b_norm = b_sync / np.maximum(b_sync.max(axis=0, keepdims=True), 1e-9)

    silent = r_sync < max(1e-4, 0.04 * np.percentile(r_sync, 90))

    # 5. key
    tonic, mode = _estimate_key(c_norm.mean(axis=1))
    diatonic = _diatonic_set(tonic, mode)

    # 6. emission scores
    types = VOCABULARIES.get(vocab, VOCABULARIES["standard"])
    labels, templ, roots, fams = _build_templates(types, seventh_weight, overtones=(feat.get('mode') != 'nnls'))
    score = c_norm.T @ templ.T                           # cosine similarity (T, S)
    score = score + bass_weight * b_norm.T[:, roots]     # bass note supports its chord
    in_key = np.array([(r, f) in diatonic for r, f in zip(roots, fams)])
    score = score + key_bonus * in_key[None, :]
    # extended chords (7ths, 6ths, sus...) must earn their place: small penalty vs. plain triads
    is_ext = np.array([lab_type not in ("maj", "min") for _ in range(12) for lab_type in types])
    score = score - ext_penalty * is_ext[None, :]
    emit(0.68, "Chord templates matched against harmonic and bass features")

    # "N" (no chord) state
    n_col = np.where(silent, 2.0, -1.0)[:, None]
    score = np.hstack([score, n_col])
    labels = labels + ["N"]
    S = score.shape[1]

    z = beta * score
    log_emit = z - np.logaddexp.reduce(z, axis=1, keepdims=True)

    # 7. transitions: strong self-loop, the rest uniform
    trans = np.full((S, S), (1 - p_stay) / (S - 1))
    np.fill_diagonal(trans, p_stay)
    path = _viterbi(log_emit, np.log(trans))
    emit(0.78, "Viterbi sequence decoding completed")

    # 8. build segments
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    times[0], times[-1] = 0.0, duration
    assert len(times) == T + 1, (len(times), T)

    spans = []
    for i, state in enumerate(path):
        if spans and spans[-1][2] == state:
            spans[-1] = (spans[-1][0], i + 1, int(state))
        else:
            spans.append((i, i + 1, int(state)))

    # Merge very short spans before calculating confidence, so the reported
    # probability describes the final displayed interval rather than the
    # fragment that was absorbed into its neighbour.
    if min_segment > 0 and len(spans) > 1:
        changed = True
        while changed and len(spans) > 1:
            changed = False
            for i, (a0, a1, state) in enumerate(spans):
                if float(times[a1] - times[a0]) >= min_segment:
                    continue
                neighbor_i = i - 1 if i > 0 else i + 1
                b0, b1, neighbor_state = spans[neighbor_i]
                merged = (min(a0, b0), max(a1, b1), neighbor_state)
                spans[neighbor_i] = merged
                spans.pop(i)
                # Adjacent spans may now have the same state.
                compact = []
                for part in spans:
                    if compact and compact[-1][2] == part[2]:
                        compact[-1] = (compact[-1][0], part[1], part[2])
                    else:
                        compact.append(part)
                spans = compact
                changed = True
                break

    segs: List[Segment] = []
    frame_probs = np.exp(log_emit)
    for start_i, end_i, state in spans:
        name = labels[state]
        mean_probs = frame_probs[start_i:end_i].mean(axis=0)
        confidence = float(np.clip(mean_probs[state], 0.0, 1.0))
        order = np.argsort(mean_probs)[::-1]
        alternatives = []
        for candidate_idx in order:
            candidate_name = labels[int(candidate_idx)]
            if candidate_name == name or candidate_name == "N":
                continue
            alternatives.append({"chord": candidate_name, "confidence": float(mean_probs[candidate_idx])})
            if len(alternatives) >= 3:
                break
        segs.append(Segment(float(times[start_i]), float(times[end_i]), name, confidence, alternatives))

    if slash:  # inversions: label X/bass when the bass clearly plays another tone of the chord
        info = {NOTES[r] + CHORD_TYPES[t][1]: (r, {iv % 12 for iv in CHORD_TYPES[t][0]})
                for r in range(12) for t in types}
        for sg in segs:
            if sg.chord not in info:
                continue
            idx = [i for i in range(T) if times[i] >= sg.start - 1e-6 and times[i + 1] <= sg.end + 1e-6]
            if not idx:
                continue
            bv = b_norm[:, idx].mean(axis=1)
            bpc = int(bv.argmax())
            root, tones = info[sg.chord]
            if bpc != root and (bpc - root) % 12 in tones and bv[bpc] > 0.6 and bv[bpc] > 1.8 * bv[root]:
                sg.chord = f"{sg.chord}/{NOTES[bpc]}"

    if refine and len(segs) > 1:
        _, templ_ov, _, _ = _build_templates(types, seventh_weight, overtones=True)
        lab2i = {lab: i for i, lab in enumerate(labels[:-1])}
        segs = _refine_boundaries(segs, feat, templ_ov, lab2i, float(np.mean(np.diff(times))), snap_onsets)

    emit(1.0, f"{len(segs)} chord segments finalized")
    # ``tuning`` is expressed in fractions of a semitone (12 bins/octave).
    tuning_cents = tuning * 100.0
    return Result(segs, tempo, f"{NOTES[tonic]} {mode}", tuning_cents, duration,
                  [float(b) for b in feat["beat_times"]],
                  [float(librosa.frames_to_time(f, sr=sr, hop_length=hop))
                   for f in feat.get("onset_frames", [])])


def _refine_boundaries(segs, feat, templ_ov, lab2i, mean_beat, snap_onsets=True, snap_window=0.08):
    """Move every chord boundary from the beat grid to the exact frame (~23 ms) where the harmony changes.
    Change-point search: inside +-1 beat around the boundary choose the split that best explains the frames
    as 'chord A before, chord B after' (max-likelihood), using notes C3..B6 (short CQT filters = sharp timing)."""
    Cfr, sr, hop = feat["Cfr"], feat["sr"], feat["hop"]
    onsets = np.asarray(feat.get("onset_frames", []), dtype=float) * hop / sr
    F = Cfr.shape[1]
    ns = np.arange(12, 60)
    E = np.maximum.reduce([Cfr[3 * ns - 1], Cfr[3 * ns], Cfr[3 * ns + 1]])        # (48, F)
    chroma = np.zeros((12, F), dtype=np.float32)
    for k, n in enumerate(ns):
        chroma[n % 12] += E[k]
    chroma /= np.maximum(np.linalg.norm(chroma, axis=0, keepdims=True), 1e-9)
    for i in range(len(segs) - 1):
        a, b = segs[i], segs[i + 1]
        ia, ib = lab2i.get(a.chord.split("/")[0]), lab2i.get(b.chord.split("/")[0])
        if ia is None or ib is None or ia == ib:
            continue
        t0 = a.end
        W = max(0.1, min(mean_beat, 0.45 * min(a.end - a.start, b.end - b.start)))
        f0 = max(int((t0 - W) * sr / hop), 0)
        f1 = min(int((t0 + W) * sr / hop) + 1, F)
        if f1 - f0 < 3:
            continue
        seg = chroma[:, f0:f1]
        sc = templ_ov[ia] @ seg - templ_ov[ib] @ seg           # >0: frame looks like A, <0: like B
        sc = np.convolve(sc, np.ones(3) / 3, mode="same")
        cs = np.concatenate([[0.0], np.cumsum(sc)])
        k = int(np.argmax(2 * cs - cs[-1]))                      # best split point
        t_new = (f0 + k - 0.5) * hop / sr
        if snap_onsets and len(onsets):          # real chord changes are played on a note attack -> snap to it
            j = int(np.argmin(np.abs(onsets - t_new)))
            if abs(onsets[j] - t_new) <= snap_window:
                t_new = float(onsets[j])
        t_new = float(min(max(t_new, a.start + 0.05), b.end - 0.05))
        a.end = b.start = t_new
    return segs


def _merge_short(segs: List[Segment], min_len: float) -> List[Segment]:
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, s in enumerate(segs):
            if s.end - s.start < min_len:
                j = i - 1 if i > 0 else i + 1
                n = segs[j]
                n.start, n.end = min(n.start, s.start), max(n.end, s.end)
                segs.pop(i)
                changed = True
                break
    out = []
    for s in segs:
        if out and out[-1].chord == s.chord:
            out[-1].end = s.end
        else:
            out.append(s)
    return out


def analyze(path: Optional[str] = None, vocab: str = "simple", max_seconds: Optional[float] = 600,
            samples: Optional[np.ndarray] = None, sr: int = 22050,
            progress: Optional[ProgressFn] = None, **params) -> Result:
    mode = params.pop("mode", "nnls")
    feat = compute_features(path, max_seconds=max_seconds, sr=sr, samples=samples,
                            mode=mode, progress=progress)
    return decode(feat, vocab=vocab, progress=progress, **params)


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("usage: python chord_engine.py song.mp3 [simple|standard|full]")
        raise SystemExit(1)
    r = analyze(sys.argv[1], vocab=sys.argv[2] if len(sys.argv) > 2 else "standard")
    print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
