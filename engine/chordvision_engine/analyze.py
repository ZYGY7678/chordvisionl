"""The analysis pipeline: audio file in, chord sequence out."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Dict, Any

import numpy as np

from . import audio as audio_io
from .cqt import cqt, harmonic_percussive_split, CQTSpec
from .chroma import (
    chromagram, bass_chromagram, estimate_tuning, frame_energy, normalize_frames,
)
from .theory import (
    build_templates, chord_name, chord_pitch_classes, pc_name,
    QUALITY_SETS, NO_CHORD,
)
from .decode import (
    DecoderConfig, decode_chords, path_to_segments, merge_short_segments,
)
from .rhythm import (
    onset_strength, pick_onsets, estimate_tempo, track_beats, snap_to_onsets,
)
from .harmony import (
    estimate_key, estimate_key_from_chords, detect_bass, chord_pitch_histogram,
)
from .types import ChordEvent, ChordCandidate, AnalysisResult, KeyEstimate

ProgressFn = Callable[[str, float, str], None]
"""Called as ``progress(stage, fraction_0_to_1, human_readable_message)``."""


@dataclass
class EngineConfig:
    """Analysis parameters. The defaults are what the server uses."""
    sample_rate: int = 11025
    hop_length: int = 512
    f_min: float = 32.70319566257483      # C1
    bins_per_octave: int = 36
    n_octaves: int = 7

    use_hpss: bool = True
    """Separate out percussion before building the chromagram.  Drums put
    broadband energy on exactly the frames where chords change, which is
    the worst possible place for noise."""

    chroma_gamma: float = 100.0
    chroma_smooth_frames: int = 13
    """Median filter length for the chromagram, about 0.6 s at the default
    frame rate.  This removes passing notes without blurring real chord
    boundaries, because a median filter preserves step edges."""

    estimate_tuning_enabled: bool = True
    snap_to_onsets_enabled: bool = True
    snap_window_s: float = 0.10

    detect_bass_enabled: bool = True
    use_flats: bool = False

    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    max_duration_s: Optional[float] = None
    """Optional cap, for very long uploads."""


class ChordEngine:
    """Reusable analyser. Holding one instance avoids rebuilding templates."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()

    # -- public API ------------------------------------------------------

    def analyze_file(self, path: str, progress: Optional[ProgressFn] = None) -> AnalysisResult:
        cfg = self.config
        self._emit(progress, "prepare_audio", 0.0, "Decoding the audio file")

        probe = audio_io.probe(path)
        samples, sr = audio_io.decode(
            path,
            target_sr=cfg.sample_rate,
            mono=True,
            max_seconds=cfg.max_duration_s,
        )
        self._emit(progress, "prepare_audio", 1.0, "Audio decoded")

        result = self.analyze_samples(samples, sr, progress=progress)
        result.diagnostics["source"] = {
            "container": probe.format_name,
            "codec": probe.codec_name,
            "originalSampleRate": probe.sample_rate,
            "originalChannels": probe.channels,
            "probedDurationS": round(probe.duration_s, 3),
        }
        return result

    def analyze_samples(
        self,
        y: np.ndarray,
        sr: int,
        progress: Optional[ProgressFn] = None,
    ) -> AnalysisResult:
        cfg = self.config
        t_start = time.time()

        if y.ndim != 1:
            raise ValueError("analyze_samples expects a mono signal")
        if y.size == 0:
            raise ValueError("the audio contains no samples")

        duration_ms = int(round(1000.0 * y.size / sr))
        loudness_db = audio_io.rms_db(y)

        y = audio_io.peak_normalize(y)

        # ---- Constant-Q Transform -------------------------------------
        self._emit(progress, "musical_analysis", 0.05, "Running the constant-Q transform")
        n_bins = cfg.bins_per_octave * cfg.n_octaves
        C, spec = cqt(
            y, sr,
            f_min=cfg.f_min,
            n_bins=n_bins,
            bins_per_octave=cfg.bins_per_octave,
            hop_length=cfg.hop_length,
        )
        n_frames = C.shape[1]
        frame_dur_s = cfg.hop_length / sr

        # ---- energy & onsets ------------------------------------------
        energy = frame_energy(C, spec)
        env = onset_strength(C)
        onsets = pick_onsets(env, spec.frame_rate_hz)
        tempo = estimate_tempo(env, spec.frame_rate_hz)
        beats = track_beats(env, spec.frame_rate_hz, tempo)
        self._emit(progress, "musical_analysis", 0.45, "Onsets and tempo estimated")

        # ---- percussion removal ---------------------------------------
        C_harm = C
        if cfg.use_hpss and n_frames > 32:
            C_harm, _C_perc = harmonic_percussive_split(C)
        self._emit(progress, "musical_analysis", 0.7, "Separating harmonic content")

        # ---- tuning ----------------------------------------------------
        tuning_cents = 0.0
        if cfg.estimate_tuning_enabled:
            tuning_cents = estimate_tuning(C_harm, spec)
        self._emit(progress, "musical_analysis", 0.85,
                   f"Tuning offset {tuning_cents:+.1f} cents")

        # ---- chroma -----------------------------------------------------
        chroma = chromagram(
            C_harm, spec,
            tuning_cents=tuning_cents,
            gamma=cfg.chroma_gamma,
            use_octave_weighting=True,
            smooth_frames=cfg.chroma_smooth_frames,
            normalize="l2",
        )
        bass_chroma = bass_chromagram(
            C_harm, spec,
            tuning_cents=tuning_cents,
            gamma=cfg.chroma_gamma,
            smooth_frames=max(3, cfg.chroma_smooth_frames // 2),
        )
        self._emit(progress, "chord_detection", 0.1, "Chromagram built")

        # ---- decode ------------------------------------------------------
        decoded = decode_chords(
            chroma,
            frame_duration_s=frame_dur_s,
            energy=energy,
            cfg=cfg.decoder,
            bass_chroma=bass_chroma,
        )
        self._emit(progress, "chord_detection", 0.6, "Chord sequence decoded")

        segments = path_to_segments(decoded.path)
        min_frames = max(1, int(round(cfg.decoder.min_segment_s / frame_dur_s)))
        segments = merge_short_segments(segments, min_frames, decoded.posterior)

        # ---- snap boundaries to onsets -----------------------------------
        if cfg.snap_to_onsets_enabled and onsets.size and len(segments) > 1:
            segments = self._snap_segments(
                segments, onsets, int(round(cfg.snap_window_s / frame_dur_s)), n_frames
            )

        # ---- build chord events ------------------------------------------
        T, labels, priors = build_templates(QUALITY_SETS[cfg.decoder.quality_set])
        sim = T @ chroma
        log_prior = np.log(np.maximum(priors, 1e-12))

        events: List[ChordEvent] = []
        for start, end, state in segments:
            events.append(self._make_event(
                start, end, state, decoded, sim, log_prior, labels,
                bass_chroma, frame_dur_s, n_frames,
            ))

        self._emit(progress, "chord_detection", 0.9, "Chord events assembled")

        # ---- key ----------------------------------------------------------
        from .theory import name_to_pc
        chord_spans = [
            ((e.end_time_ms - e.start_time_ms) / 1000.0,
             name_to_pc(e.root), e.quality)
            for e in events if e.quality != NO_CHORD
        ]
        key = estimate_key_from_chords(
            chord_spans,
            chroma_hist=(chroma @ energy) if energy.size == chroma.shape[1] else None,
        )
        self._emit(progress, "chord_detection", 1.0, f"Key: {key.name}")

        elapsed = time.time() - t_start
        audio_s = y.size / sr

        return AnalysisResult(
            chords=events,
            key=key,
            duration_ms=duration_ms,
            sample_rate=sr,
            tuning_offset_cents=tuning_cents,
            beat_times_ms=[int(round(b * frame_dur_s * 1000)) for b in beats.tolist()],
            onset_times_ms=[int(round(o * frame_dur_s * 1000)) for o in onsets.tolist()],
            tempo_bpm=tempo,
            frame_rate_hz=spec.frame_rate_hz,
            diagnostics={
                "frames": int(n_frames),
                "frameDurationMs": round(frame_dur_s * 1000, 2),
                "qualitySet": cfg.decoder.quality_set,
                "chordStates": int(T.shape[0]) + 1,
                "inputLoudnessDbfs": None if np.isneginf(loudness_db) else round(loudness_db, 2),
                "onsetCount": int(onsets.size),
                "beatCount": int(beats.size),
                "analysisSeconds": round(elapsed, 3),
                "realtimeFactor": round(audio_s / elapsed, 2) if elapsed > 0 else None,
                "engineVersion": "1.0.0",
            },
        )

    # -- internals -------------------------------------------------------

    @staticmethod
    def _emit(progress: Optional[ProgressFn], stage: str, frac: float, msg: str) -> None:
        if progress is not None:
            progress(stage, float(np.clip(frac, 0.0, 1.0)), msg)

    @staticmethod
    def _snap_segments(segments, onsets, window_frames, n_frames):
        if window_frames <= 0:
            return segments
        boundaries = np.array([s[0] for s in segments[1:]], dtype=np.int64)
        snapped = snap_to_onsets(boundaries, onsets, window_frames)

        out = []
        prev_start = segments[0][0]
        for i, (start, end, state) in enumerate(segments):
            new_start = prev_start if i == 0 else int(snapped[i - 1])
            new_end = int(snapped[i]) if i < len(snapped) else n_frames
            if new_end <= new_start:
                new_end = new_start + 1
            out.append((new_start, new_end, state))
            prev_start = new_end

        # Drop anything the snapping collapsed to nothing.
        return [s for s in out if s[1] > s[0]]

    def _make_event(
        self, start, end, state, decoded, sim, log_prior, labels,
        bass_chroma, frame_dur_s, n_frames,
    ) -> ChordEvent:
        cfg = self.config
        start_ms = int(round(start * frame_dur_s * 1000))
        end_ms = int(round(min(end, n_frames) * frame_dur_s * 1000))

        if state == decoded.no_chord_index:
            return ChordEvent(
                chord_name=NO_CHORD, root=NO_CHORD, quality=NO_CHORD,
                bass_note=None, start_time_ms=start_ms, end_time_ms=end_ms,
                confidence=float(np.mean(decoded.posterior[state, start:end])),
                alternatives=[],
            )

        root_pc, quality = labels[state]

        # Segment-level posterior. Averaging the similarity across the whole
        # segment before softmaxing is much steadier than averaging the
        # per-frame posteriors: a chord is a property of the span, not of
        # any single 46 ms slice.
        seg_sim = sim[:, start:end].mean(axis=1)
        seg_logp = cfg.decoder.kappa * seg_sim + log_prior
        seg_logp -= seg_logp.max()
        seg_post = np.exp(seg_logp)
        seg_post /= max(seg_post.sum(), 1e-300)

        confidence = float(seg_post[state])

        order = np.argsort(seg_post)[::-1]
        alternatives: List[ChordCandidate] = []
        for idx in order:
            if int(idx) == state:
                continue
            r, q = labels[int(idx)]
            alternatives.append(ChordCandidate(
                chord_name=chord_name(r, q, None, cfg.use_flats),
                confidence=float(seg_post[int(idx)]),
            ))
            if len(alternatives) >= cfg.decoder.n_alternatives:
                break

        bass_pc = None
        if cfg.detect_bass_enabled:
            pcs = chord_pitch_classes(root_pc, quality)
            bass_pc, _bass_conf = detect_bass(
                bass_chroma, start, min(end, n_frames), pcs, root_pc
            )

        return ChordEvent(
            chord_name=chord_name(root_pc, quality, bass_pc, cfg.use_flats),
            root=pc_name(root_pc, cfg.use_flats),
            quality=quality,
            bass_note=pc_name(bass_pc, cfg.use_flats) if bass_pc is not None else None,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
            confidence=confidence,
            alternatives=alternatives,
        )

    @staticmethod
    def _event_spans(events: List[ChordEvent], frame_dur_s: float):
        from .theory import name_to_pc
        out = []
        for e in events:
            if e.quality == NO_CHORD:
                continue
            s = int(round(e.start_time_ms / 1000.0 / frame_dur_s))
            t = int(round(e.end_time_ms / 1000.0 / frame_dur_s))
            out.append((s, t, name_to_pc(e.root), e.quality))
        return out


def analyze_file(path: str, config: Optional[EngineConfig] = None,
                 progress: Optional[ProgressFn] = None) -> AnalysisResult:
    """Convenience wrapper: analyse one file with a fresh engine."""
    return ChordEngine(config).analyze_file(path, progress)


def analyze_samples(y: np.ndarray, sr: int, config: Optional[EngineConfig] = None,
                    progress: Optional[ProgressFn] = None) -> AnalysisResult:
    """Convenience wrapper: analyse an in-memory mono signal."""
    return ChordEngine(config).analyze_samples(y, sr, progress)
