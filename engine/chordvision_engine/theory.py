"""Music theory: pitch classes, chord qualities, templates and naming.

The chord templates built here are the model the recogniser matches the
audio against, so how they are built decides how well the whole engine
works.

A naive template marks the chord tones with 1 and everything else with 0:
C major becomes [1,0,0,0,1,0,0,1,0,0,0,0].  That model is wrong about real
instruments.  A piano playing C also radiates energy at the harmonics of
each note: C's third harmonic is a G, its fifth harmonic is an E, and its
seventh is close to a Bb.  So a plain C major chord genuinely puts energy
into pitch classes that are not in the triad, and a binary template
penalises the correct answer for it.

The templates below are therefore built by summing a decaying harmonic
series for every chord tone, which is what a real instrument produces.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Iterable
import re

import numpy as np

PITCH_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCH_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Name -> pitch class, accepting every spelling we might be handed.
_NAME_TO_PC: Dict[str, int] = {}
for _i, _n in enumerate(PITCH_SHARP):
    _NAME_TO_PC[_n] = _i
for _i, _n in enumerate(PITCH_FLAT):
    _NAME_TO_PC[_n] = _i
_NAME_TO_PC.update({
    "B#": 0, "Cb": 11, "E#": 5, "Fb": 4,
    "C##": 2, "Dbb": 0, "F##": 7, "Gbb": 5,
})


# Intervals in semitones above the root.
CHORD_QUALITIES: Dict[str, Tuple[int, ...]] = {
    "maj":   (0, 4, 7),
    "min":   (0, 3, 7),
    "dim":   (0, 3, 6),
    "aug":   (0, 4, 8),
    "sus2":  (0, 2, 7),
    "sus4":  (0, 5, 7),
    "7":     (0, 4, 7, 10),
    "maj7":  (0, 4, 7, 11),
    "min7":  (0, 3, 7, 10),
    "dim7":  (0, 3, 6, 9),
    "m7b5":  (0, 3, 6, 10),
    "6":     (0, 4, 7, 9),
    "min6":  (0, 3, 7, 9),
    "add9":  (0, 2, 4, 7),
    "9":     (0, 2, 4, 7, 10),
    "maj9":  (0, 2, 4, 7, 11),
    "min9":  (0, 2, 3, 7, 10),
    # An eleventh chord normally drops the third, which would clash a
    # semitone below the eleventh itself.
    "11":    (0, 2, 5, 7, 10),
    # A thirteenth normally drops the eleventh for the same reason.
    "13":    (0, 2, 4, 7, 9, 10),
}

# How the quality is spelled after the root in the chord symbol.
QUALITY_SUFFIX: Dict[str, str] = {
    "maj": "",     "min": "m",      "dim": "dim",   "aug": "aug",
    "sus2": "sus2", "sus4": "sus4", "7": "7",       "maj7": "maj7",
    "min7": "m7",  "dim7": "dim7",  "m7b5": "m7b5", "6": "6",
    "min6": "m6",  "add9": "add9",  "9": "9",       "maj9": "maj9",
    "min9": "m9",  "11": "11",      "13": "13",
}

# Prior probability of each quality appearing at all, before any audio is
# seen.  These are not tuned parameters pulled out of the air: they encode
# the uncontroversial fact that plain triads vastly outnumber thirteenth
# chords in recorded music.  Without a prior the decoder will happily
# explain any messy frame with a six-note chord, because a six-note
# template can fit almost any chroma vector.
QUALITY_PRIOR: Dict[str, float] = {
    "maj": 1.00, "min": 0.95,
    "7": 0.40,   "min7": 0.35, "maj7": 0.28,
    "sus4": 0.20, "sus2": 0.14,
    "6": 0.10,   "min6": 0.07,
    "dim": 0.07, "aug": 0.04,
    "m7b5": 0.05, "dim7": 0.04,
    "add9": 0.09, "9": 0.07, "maj9": 0.05, "min9": 0.05,
    "11": 0.03,  "13": 0.03,
}

# Reduced sets, useful for evaluation and for users who want plain output.
QUALITY_SETS: Dict[str, Tuple[str, ...]] = {
    "triads": ("maj", "min"),
    "triads_ext": ("maj", "min", "dim", "aug", "sus2", "sus4"),
    "sevenths": ("maj", "min", "7", "min7", "maj7", "dim", "aug", "sus2", "sus4", "m7b5", "dim7"),
    "full": tuple(CHORD_QUALITIES.keys()),
}

NO_CHORD = "N"


def pc_name(pc: int, use_flats: bool = False) -> str:
    """Pitch class number to note name."""
    table = PITCH_FLAT if use_flats else PITCH_SHARP
    return table[int(pc) % 12]


def name_to_pc(name: str) -> int:
    """Note name to pitch class number. Raises ValueError on nonsense."""
    key = name.strip()
    if not key:
        raise ValueError("empty note name")
    key = key[0].upper() + key[1:]
    if key not in _NAME_TO_PC:
        raise ValueError(f"unrecognised note name: {name!r}")
    return _NAME_TO_PC[key]


def chord_name(root_pc: int, quality: str, bass_pc: Optional[int] = None,
               use_flats: bool = False) -> str:
    """Assemble a display chord symbol, e.g. ``C``, ``Am7``, ``G/B``."""
    if quality == NO_CHORD:
        return NO_CHORD
    if quality not in QUALITY_SUFFIX:
        raise ValueError(f"unknown chord quality: {quality!r}")
    base = pc_name(root_pc, use_flats) + QUALITY_SUFFIX[quality]
    if bass_pc is not None and int(bass_pc) % 12 != int(root_pc) % 12:
        base += "/" + pc_name(bass_pc, use_flats)
    return base


_CHORD_RE = re.compile(
    r"^(?P<root>[A-Ga-g](?:#|b|##|bb)?)"
    r"(?P<qual>maj7|maj9|m7b5|min7|min9|dim7|sus2|sus4|add9|dim|aug|m7|m9|m6|maj|min|13|11|9|7|6|m)?"
    r"(?:/(?P<bass>[A-Ga-g](?:#|b)?))?$"
)

_SUFFIX_TO_QUALITY = {
    "": "maj", "maj": "maj", "M": "maj",
    "m": "min", "min": "min", "-": "min",
    "dim": "dim", "aug": "aug",
    "sus2": "sus2", "sus4": "sus4",
    "7": "7", "maj7": "maj7", "m7": "min7", "min7": "min7",
    "dim7": "dim7", "m7b5": "m7b5",
    "6": "6", "m6": "min6",
    "add9": "add9", "9": "9", "maj9": "maj9", "m9": "min9", "min9": "min9",
    "11": "11", "13": "13",
}


def parse_chord(symbol: str) -> Tuple[int, str, Optional[int]]:
    """Parse a chord symbol into ``(root_pc, quality, bass_pc_or_None)``.

    Used by the editor, by the importers and by the evaluation harness so
    that a chord written by a human and a chord produced by the engine are
    compared as the same object rather than as two strings.
    """
    s = symbol.strip()
    if s in (NO_CHORD, "", "-", "NC", "N.C."):
        return (-1, NO_CHORD, None)

    m = _CHORD_RE.match(s)
    if not m:
        raise ValueError(f"could not parse chord symbol: {symbol!r}")

    root = name_to_pc(m.group("root"))
    suffix = m.group("qual") or ""
    if suffix not in _SUFFIX_TO_QUALITY:
        raise ValueError(f"unsupported chord quality in {symbol!r}")
    quality = _SUFFIX_TO_QUALITY[suffix]
    bass = name_to_pc(m.group("bass")) if m.group("bass") else None
    return (root, quality, bass)


def transpose_chord(symbol: str, semitones: int, use_flats: bool = False) -> str:
    """Transpose a chord symbol by ``semitones``, keeping any slash bass."""
    if symbol.strip() in (NO_CHORD, ""):
        return symbol
    root, quality, bass = parse_chord(symbol)
    new_root = (root + semitones) % 12
    new_bass = (bass + semitones) % 12 if bass is not None else None
    return chord_name(new_root, quality, new_bass, use_flats)


def chord_pitch_classes(root_pc: int, quality: str) -> List[int]:
    """The pitch classes a chord contains."""
    return sorted({(root_pc + i) % 12 for i in CHORD_QUALITIES[quality]})


def harmonic_profile(
    pitch_classes: Iterable[int],
    n_harmonics: int = 5,
    decay: float = 0.45,
    root_boost: float = 1.0,
    root_pc: Optional[int] = None,
    tone_weight: float = 0.45,
) -> np.ndarray:
    """Expected pitch class energy for a set of notes played by real instruments.

    Each note contributes its own harmonic series.  Harmonic *h* of a note
    lands ``12*log2(h)`` semitones above it, which for h = 1..6 is the
    unison, the octave, the fifth, the second octave and the major third
    two octaves up.  Amplitude falls off as ``decay**(h-1)``.

    ``tone_weight`` then adds a flat bonus to every pitch class that is an
    actual member of the chord.  This matters more than it looks.  The
    third harmonic of E is B, so a plain C major chord already deposits
    energy on B - and without this term the template for C and the
    template for Cmaj7 become nearly the same vector, so the engine can
    never hear the difference between them.  Separating "this note is in
    the chord" from "this note is a by-product of the chord" keeps real
    chord tones distinguishable from their own overtones.
    """
    profile = np.zeros(12, dtype=np.float64)
    pcs = list(pitch_classes)
    for pc in pcs:
        weight = 1.0
        if root_pc is not None and pc % 12 == root_pc % 12:
            weight = root_boost
        for h in range(1, n_harmonics + 1):
            offset = int(round(12.0 * np.log2(h)))
            profile[(pc + offset) % 12] += weight * (decay ** (h - 1))

    if tone_weight:
        for pc in pcs:
            w = root_boost if (root_pc is not None and pc % 12 == root_pc % 12) else 1.0
            profile[pc % 12] += tone_weight * w
    return profile


def build_templates(
    qualities: Iterable[str] = QUALITY_SETS["full"],
    n_harmonics: int = 5,
    decay: float = 0.45,
    root_boost: float = 1.25,
    normalize: bool = True,
    tone_weight: float = 0.45,
) -> Tuple[np.ndarray, List[Tuple[int, str]], np.ndarray]:
    """Build the full template bank.

    Returns ``(T, labels, priors)``:
      * ``T``      - shape (n_states, 12), one unit-length row per chord
      * ``labels`` - ``(root_pc, quality)`` for each row
      * ``priors`` - prior weight for each row

    The "no chord" state is **not** included here; the decoder adds it,
    because it is scored against frame energy rather than against chroma
    shape.
    """
    qualities = list(qualities)
    rows: List[np.ndarray] = []
    labels: List[Tuple[int, str]] = []
    priors: List[float] = []

    for quality in qualities:
        if quality not in CHORD_QUALITIES:
            raise ValueError(f"unknown chord quality: {quality!r}")
        base_pcs = CHORD_QUALITIES[quality]
        for root in range(12):
            pcs = [(root + i) % 12 for i in base_pcs]
            prof = harmonic_profile(
                pcs, n_harmonics=n_harmonics, decay=decay,
                root_boost=root_boost, root_pc=root, tone_weight=tone_weight,
            )
            rows.append(prof)
            labels.append((root, quality))
            priors.append(QUALITY_PRIOR.get(quality, 0.05))

    T = np.vstack(rows)
    if normalize:
        norms = np.sqrt(np.sum(T * T, axis=1, keepdims=True))
        T = T / np.maximum(norms, 1e-12)
    return T, labels, np.asarray(priors, dtype=np.float64)


# Krumhansl-Kessler key profiles, from experiments where listeners rated
# how well each pitch class fitted an established key.  They are the
# standard reference profiles for key finding.
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
                     5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
                     4.75, 3.98, 2.69, 3.34, 3.17])


def circle_of_fifths_distance(pc_a: int, pc_b: int) -> int:
    """Steps around the circle of fifths between two pitch classes (0..6).

    Chords whose roots are close on the circle of fifths share more notes
    and follow each other far more often than distant ones, which is what
    the transition model uses.
    """
    # Position on the circle: multiply by 7 (a fifth) modulo 12.
    a = (pc_a * 7) % 12
    b = (pc_b * 7) % 12
    d = abs(a - b) % 12
    return min(d, 12 - d)
