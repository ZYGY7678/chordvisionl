"""Render audio with a chord progression that is known exactly.

This exists so the engine can be measured instead of described.  Every
rendered clip comes with a frame-accurate list of which chord is sounding
when, because we are the ones who put it there.

What this does and does not prove
---------------------------------
It proves the engine recovers harmony from a real acoustic signal: the
renderer writes nothing but samples, and the analyser reads nothing but
samples.  Pitch detection, tuning, octave folding, template matching,
sequence decoding, boundary placement and inversion detection are all
exercised end to end.

It does **not** prove the engine performs the same on commercial
recordings.  Real mixes add distorted guitars with shifting spectra,
vocals that glide between pitches, heavy compression, and production
choices that blur the harmony on purpose.  Synthetic material is
genuinely easier.  The numbers measured here are an upper bound on real
world performance, and they are reported as such.

The renderer therefore works hard to avoid being trivially easy:
instruments have realistic harmonic spectra and decay, notes are voiced
across octaves and inverted rather than stacked in root position, strums
are spread in time, a bass line walks, drums put broadband noise exactly
on the chord changes, and the whole mix passes through a reverb.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Sequence, Dict

import numpy as np

A4_HZ = 440.0
A4_MIDI = 69


def midi_to_hz(midi: float, tuning_cents: float = 0.0) -> float:
    return A4_HZ * (2.0 ** ((midi - A4_MIDI) / 12.0)) * (2.0 ** (tuning_cents / 1200.0))


@dataclass
class ChordSpan:
    """One chord in the ground truth, with exact boundaries in seconds."""
    symbol: str
    start_s: float
    end_s: float
    root_pc: int
    quality: str
    bass_pc: Optional[int] = None


# ---------------------------------------------------------------------------
# Instrument voices
# ---------------------------------------------------------------------------

#: Relative amplitude of harmonics 1..N for a few instrument characters.
#: These are rough but realistic shapes: a piano's energy falls away
#: steadily, a clean electric guitar keeps more upper harmonics, an organ
#: emphasises octaves and fifths, and a bass is nearly a pure fundamental.
TIMBRES: Dict[str, np.ndarray] = {
    "piano":  np.array([1.00, 0.46, 0.28, 0.17, 0.11, 0.07, 0.05, 0.035, 0.02, 0.015]),
    "guitar": np.array([1.00, 0.62, 0.40, 0.30, 0.21, 0.16, 0.12, 0.090, 0.07, 0.050]),
    "organ":  np.array([1.00, 0.72, 0.52, 0.45, 0.14, 0.30, 0.08, 0.220, 0.05, 0.040]),
    "epiano": np.array([1.00, 0.30, 0.52, 0.12, 0.22, 0.06, 0.09, 0.040, 0.02, 0.010]),
    "bass":   np.array([1.00, 0.52, 0.21, 0.09, 0.05, 0.03, 0.02, 0.010, 0.00, 0.000]),
    "strings":np.array([1.00, 0.58, 0.44, 0.33, 0.26, 0.20, 0.15, 0.110, 0.08, 0.060]),
}


def _adsr(n: int, sr: int, attack: float, decay: float,
          sustain: float, release: float) -> np.ndarray:
    """A plain ADSR envelope, sample accurate."""
    env = np.zeros(n, dtype=np.float64)
    a = min(n, int(attack * sr))
    d = min(n - a, int(decay * sr))
    r = min(n - a - d, int(release * sr))
    s = max(0, n - a - d - r)

    pos = 0
    if a > 0:
        env[pos:pos + a] = np.linspace(0.0, 1.0, a, endpoint=False)
        pos += a
    if d > 0:
        env[pos:pos + d] = np.linspace(1.0, sustain, d, endpoint=False)
        pos += d
    if s > 0:
        env[pos:pos + s] = sustain
        pos += s
    if r > 0:
        env[pos:pos + r] = np.linspace(sustain, 0.0, r)
    return env


def render_note(
    midi: float,
    duration_s: float,
    sr: int,
    timbre: str = "piano",
    amplitude: float = 0.3,
    tuning_cents: float = 0.0,
    inharmonicity: float = 0.0004,
    attack: float = 0.006,
    decay: float = 0.25,
    sustain: float = 0.55,
    release: float = 0.35,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Render a single note as a sum of (slightly inharmonic) partials.

    Real strings are stiff, so their partials sit progressively sharp of
    exact integer multiples.  Including that makes the test material
    behave more like an instrument and less like a textbook example - it
    is one of the things that smears a chromagram in practice.
    """
    rng = rng or np.random.default_rng(0)
    n = max(1, int(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / sr

    partials = TIMBRES.get(timbre, TIMBRES["piano"])
    f0 = midi_to_hz(midi, tuning_cents)

    out = np.zeros(n, dtype=np.float64)
    nyq = sr * 0.5
    for i, amp in enumerate(partials):
        if amp <= 0:
            continue
        h = i + 1
        f = f0 * h * np.sqrt(1.0 + inharmonicity * h * h)
        if f >= nyq * 0.95:
            break
        phase = rng.uniform(0, 2 * np.pi)
        # Higher partials die away faster, as they do on a real instrument.
        hdecay = np.exp(-t * (0.6 + 0.55 * i))
        out += amp * hdecay * np.sin(2 * np.pi * f * t + phase)

    env = _adsr(n, sr, attack, decay, sustain, release)
    return (amplitude * env * out).astype(np.float64)


# ---------------------------------------------------------------------------
# Voicing
# ---------------------------------------------------------------------------

def voice_chord(
    root_pc: int,
    intervals: Sequence[int],
    bass_pc: Optional[int] = None,
    centre_midi: int = 60,
    spread: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[int], int]:
    """Lay a chord out as actual MIDI notes, the way a player would.

    Nobody plays every chord in root position in one octave.  Notes get
    doubled, inverted, dropped and spread across two octaves.  This
    matters for the test: a root-position close voicing makes the root the
    loudest pitch class every time, which would let a weak engine look
    strong.

    Returns ``(chord_notes, bass_note)``.
    """
    rng = rng or np.random.default_rng(0)

    pcs = sorted({(root_pc + i) % 12 for i in intervals})
    bass_target = bass_pc if bass_pc is not None else root_pc

    # Bass note, two octaves below the chord.
    bass_midi = centre_midi - 24
    bass_midi += (bass_target - bass_midi) % 12
    if bass_midi > centre_midi - 12:
        bass_midi -= 12

    notes: List[int] = []
    for pc in pcs:
        m = centre_midi + ((pc - centre_midi) % 12)
        notes.append(int(m))

    if spread:
        # Invert: push some of the lower notes up an octave, as a real
        # voicing would, so the root is not automatically at the bottom.
        n_up = int(rng.integers(0, max(1, len(notes))))
        order = rng.permutation(len(notes))[:n_up]
        for idx in order:
            notes[idx] += 12
        # Occasionally double the top note an octave higher.
        if rng.random() < 0.45 and notes:
            notes.append(max(notes) + 12)

    return sorted(notes), int(bass_midi)


# ---------------------------------------------------------------------------
# Percussion and room
# ---------------------------------------------------------------------------

def render_kick(sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(0.22 * sr)
    t = np.arange(n) / sr
    f = 120.0 * np.exp(-t * 28.0) + 45.0
    body = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 14.0)
    click = rng.normal(0, 1, n) * np.exp(-t * 180.0) * 0.25
    return (0.85 * body + click).astype(np.float64)


def render_snare(sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(0.20 * sr)
    t = np.arange(n) / sr
    noise = rng.normal(0, 1, n) * np.exp(-t * 20.0)
    tone = np.sin(2 * np.pi * 185.0 * t) * np.exp(-t * 26.0) * 0.4
    return (0.75 * noise + tone).astype(np.float64)


def render_hihat(sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(0.07 * sr)
    t = np.arange(n) / sr
    return (rng.normal(0, 1, n) * np.exp(-t * 85.0) * 0.5).astype(np.float64)


def reverb(x: np.ndarray, sr: int, rt60: float = 1.1, wet: float = 0.22,
           rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Convolve with an exponentially decaying noise burst.

    A crude but perfectly valid room model.  Reverb overlaps the tail of
    one chord onto the start of the next, which is one of the main reasons
    chord boundaries are hard to place on real recordings - so the test
    material should have it.
    """
    rng = rng or np.random.default_rng(1)
    n = int(rt60 * sr)
    if n < 2:
        return x
    t = np.arange(n) / sr
    ir = rng.normal(0, 1, n) * np.exp(-6.908 * t / rt60)
    ir[: int(0.01 * sr)] *= np.linspace(0, 1, int(0.01 * sr))
    ir /= np.sqrt(np.sum(ir ** 2))

    from scipy.signal import fftconvolve
    wet_sig = fftconvolve(x, ir, mode="full")[: len(x)]
    return ((1.0 - wet) * x + wet * wet_sig).astype(np.float64)


# ---------------------------------------------------------------------------
# Full track renderer
# ---------------------------------------------------------------------------

@dataclass
class RenderOptions:
    sr: int = 22050
    bpm: float = 96.0
    beats_per_chord: int = 4
    timbre: str = "piano"
    strum_spread_s: float = 0.0
    with_bass: bool = True
    walking_bass: bool = False
    with_drums: bool = True
    drum_level: float = 0.35
    with_reverb: bool = True
    reverb_rt60: float = 1.0
    reverb_wet: float = 0.2
    noise_db: float = -60.0
    tuning_cents: float = 0.0
    centre_midi: int = 60
    seed: int = 7


def render_progression(
    progression: Sequence[str],
    options: Optional[RenderOptions] = None,
    chord_intervals: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> Tuple[np.ndarray, int, List[ChordSpan]]:
    """Render a chord progression to audio plus its exact ground truth.

    ``progression`` is a list of chord symbols such as
    ``["C", "Am", "F", "G7"]``.  Returns ``(samples, sample_rate, spans)``.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine"))
    from chordvision_engine.theory import parse_chord, CHORD_QUALITIES

    opt = options or RenderOptions()
    rng = np.random.default_rng(opt.seed)
    sr = opt.sr

    chord_dur = opt.beats_per_chord * 60.0 / opt.bpm
    beat_dur = 60.0 / opt.bpm
    total_s = chord_dur * len(progression) + 1.2
    n_total = int(total_s * sr)

    mix = np.zeros(n_total, dtype=np.float64)
    spans: List[ChordSpan] = []

    for ci, symbol in enumerate(progression):
        root_pc, quality, bass_pc = parse_chord(symbol)
        intervals = CHORD_QUALITIES[quality]
        start_s = ci * chord_dur
        end_s = start_s + chord_dur
        spans.append(ChordSpan(
            symbol=symbol, start_s=start_s, end_s=end_s,
            root_pc=root_pc, quality=quality, bass_pc=bass_pc,
        ))

        notes, bass_midi = voice_chord(
            root_pc, intervals, bass_pc,
            centre_midi=opt.centre_midi, spread=True, rng=rng,
        )

        # Re-strike the chord on every beat so the harmony stays present
        # rather than decaying to nothing, which is what a real
        # accompaniment does.
        for b in range(opt.beats_per_chord):
            hit_s = start_s + b * beat_dur
            ring = chord_dur - b * beat_dur + 0.7
            level = 0.26 if b == 0 else 0.17
            for k, m in enumerate(notes):
                off = hit_s + (k * opt.strum_spread_s)
                sig = render_note(
                    m, ring, sr,
                    timbre=opt.timbre,
                    amplitude=level * rng.uniform(0.82, 1.18),
                    tuning_cents=opt.tuning_cents,
                    rng=rng,
                )
                _add(mix, sig, int(off * sr))

        if opt.with_bass:
            if opt.walking_bass:
                scale = [(root_pc + i) % 12 for i in intervals]
                for b in range(opt.beats_per_chord):
                    pc = scale[b % len(scale)] if b else (bass_pc if bass_pc is not None else root_pc)
                    m = bass_midi + ((pc - bass_midi) % 12)
                    if m > bass_midi + 8:
                        m -= 12
                    sig = render_note(
                        m, beat_dur * 1.05, sr, timbre="bass",
                        amplitude=0.34, tuning_cents=opt.tuning_cents,
                        decay=0.18, sustain=0.5, release=0.12, rng=rng,
                    )
                    _add(mix, sig, int((start_s + b * beat_dur) * sr))
            else:
                sig = render_note(
                    bass_midi, chord_dur * 0.98, sr, timbre="bass",
                    amplitude=0.38, tuning_cents=opt.tuning_cents,
                    decay=0.3, sustain=0.6, release=0.2, rng=rng,
                )
                _add(mix, sig, int(start_s * sr))

    if opt.with_drums:
        kick = render_kick(sr, rng)
        snare = render_snare(sr, rng)
        hat = render_hihat(sr, rng)
        n_beats = int(np.ceil(chord_dur * len(progression) / beat_dur))
        for b in range(n_beats):
            at = int(b * beat_dur * sr)
            if b % 4 in (0, 2):
                _add(mix, kick * opt.drum_level, at)
            if b % 4 in (1, 3):
                _add(mix, snare * opt.drum_level, at)
            for sub in (0.0, 0.5):
                _add(mix, hat * opt.drum_level * 0.55, int((b + sub) * beat_dur * sr))

    if opt.noise_db > -200:
        mix += rng.normal(0, 10 ** (opt.noise_db / 20.0), n_total)

    if opt.with_reverb:
        mix = reverb(mix, sr, opt.reverb_rt60, opt.reverb_wet, rng)

    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / peak * 0.89

    return mix.astype(np.float32), sr, spans


def _add(dst: np.ndarray, src: np.ndarray, at: int) -> None:
    """Mix ``src`` into ``dst`` at sample offset ``at``, clipping at the end."""
    if at >= len(dst):
        return
    if at < 0:
        src = src[-at:]
        at = 0
    n = min(len(src), len(dst) - at)
    if n > 0:
        dst[at:at + n] += src[:n]


def write_wav(path: str, samples: np.ndarray, sr: int) -> None:
    """Write a 16-bit PCM WAV without pulling in another dependency."""
    import wave, struct
    data = np.clip(samples, -1.0, 1.0)
    pcm = (data * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
