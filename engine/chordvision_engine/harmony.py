"""Key estimation and bass-note detection."""
from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from .theory import KK_MAJOR, KK_MINOR, pc_name, chord_pitch_classes
from .types import KeyEstimate


def _correlate_profile(histogram: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Correlation of a pitch class histogram with all 12 rotations of a profile."""
    h = histogram - histogram.mean()
    hn = np.linalg.norm(h)
    if hn <= 1e-12:
        return np.zeros(12)

    scores = np.zeros(12)
    for tonic in range(12):
        p = np.roll(profile, tonic)
        p = p - p.mean()
        pn = np.linalg.norm(p)
        if pn <= 1e-12:
            continue
        scores[tonic] = float(np.dot(h, p) / (hn * pn))
    return scores


def estimate_key(
    chroma: np.ndarray,
    weights: Optional[np.ndarray] = None,
    chord_histogram: Optional[np.ndarray] = None,
    chord_weight: float = 0.45,
) -> KeyEstimate:
    """Estimate the global key with Krumhansl-Schmuckler correlation.

    The pitch class histogram of the whole piece is correlated against the
    twenty-four rotated major and minor profiles; the best match is the
    key.

    One refinement matters in practice.  The raw chroma histogram is
    dominated by whatever is loudest and longest, which in a lot of
    arrangements is a pedal note or a busy lead line rather than the
    tonality.  Mixing in a histogram built from the *detected chords*,
    weighted by how long each chord lasts, grounds the estimate in the
    harmony instead.  Relative major and minor share the same notes and
    are separated almost entirely by which chords actually appear.
    """
    if chroma.size == 0:
        return KeyEstimate(tonic="C", mode="major", confidence=0.0, alternatives=[])

    if weights is None:
        hist = chroma.sum(axis=1)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.size != chroma.shape[1]:
            raise ValueError("weights length must match the number of frames")
        hist = chroma @ w

    total = hist.sum()
    if total > 0:
        hist = hist / total

    if chord_histogram is not None and chord_histogram.sum() > 0:
        ch = chord_histogram / chord_histogram.sum()
        hist = (1.0 - chord_weight) * hist + chord_weight * ch

    major_scores = _correlate_profile(hist, KK_MAJOR)
    minor_scores = _correlate_profile(hist, KK_MINOR)

    combined: List[Tuple[float, int, str]] = []
    for pc in range(12):
        combined.append((float(major_scores[pc]), pc, "major"))
        combined.append((float(minor_scores[pc]), pc, "minor"))
    combined.sort(reverse=True)

    best_score, best_pc, best_mode = combined[0]
    runner_up = combined[1][0]

    # Confidence is the margin over the next best candidate, squashed into
    # (0, 1).  An ambiguous piece - and plenty of real music genuinely is
    # ambiguous - gets a low number instead of a fabricated certainty.
    margin = max(0.0, best_score - runner_up)
    confidence = float(np.clip(margin / 0.12, 0.0, 1.0)) * float(np.clip(best_score, 0.0, 1.0))

    alternatives = [
        {"tonic": pc_name(pc), "mode": mode, "name": f"{pc_name(pc)} {mode}",
         "confidence": round(float(np.clip(score, 0.0, 1.0)), 4)}
        for score, pc, mode in combined[1:4]
    ]

    return KeyEstimate(
        tonic=pc_name(best_pc),
        mode=best_mode,
        confidence=round(confidence, 4),
        alternatives=alternatives,
    )


def chord_pitch_histogram(
    segments: List[Tuple[int, int, int, int, str]],
) -> np.ndarray:
    """Pitch class histogram weighted by how long each chord sounds.

    ``segments`` is a list of ``(start_frame, end_frame, root_pc, _, quality)``.
    """
    hist = np.zeros(12, dtype=np.float64)
    for start, end, root, _unused, quality in segments:
        if root < 0:
            continue
        dur = max(0, end - start)
        for pc in chord_pitch_classes(int(root), quality):
            hist[pc] += dur
    return hist


def detect_bass(
    bass_chroma: np.ndarray,
    start_frame: int,
    end_frame: int,
    chord_pcs: List[int],
    root_pc: int,
    min_margin: float = 0.16,
    min_strength: float = 0.26,
    onset_fraction: float = 0.55,
) -> Tuple[Optional[int], float]:
    """Decide the bass note of one chord segment.

    Returns ``(bass_pc_or_None, confidence)``.  ``None`` means "the bass is
    the root, or we are not confident enough to claim otherwise".

    Two rules keep this honest.  First, only a pitch class that actually
    belongs to the chord is ever reported as the bass; a passing note in
    the bass line is not an inversion.  Second, the alternative must beat
    the root by a clear margin, because reporting ``C/E`` when the real
    chord is ``C`` is a worse error than missing the inversion - a player
    reading it will finger the wrong shape.

    Only the first part of the segment is examined.  Bass players walk:
    the note under the chord change is the one that defines the inversion,
    what happens later in the bar usually does not.
    """
    if bass_chroma.size == 0 or end_frame <= start_frame:
        return None, 0.0

    span = end_frame - start_frame
    look = max(1, int(round(span * onset_fraction)))
    window = bass_chroma[:, start_frame:start_frame + look]
    if window.size == 0:
        return None, 0.0

    profile = window.mean(axis=1)
    total = profile.sum()
    if total <= 1e-9:
        return None, 0.0
    profile = profile / total

    candidates = [pc for pc in chord_pcs]
    if root_pc not in candidates:
        candidates.append(root_pc)

    scores = {pc: float(profile[pc % 12]) for pc in candidates}
    best_pc = max(scores, key=scores.get)
    best_val = scores[best_pc]
    root_val = scores.get(root_pc, 0.0)

    if best_pc % 12 == root_pc % 12:
        return None, float(min(1.0, root_val * 2.5))

    if best_val < min_strength:
        return None, 0.0
    if (best_val - root_val) < min_margin:
        return None, 0.0

    confidence = float(np.clip((best_val - root_val) / 0.35, 0.0, 1.0))
    return int(best_pc % 12), confidence


# ---------------------------------------------------------------------------
# Key estimation from the chord sequence
# ---------------------------------------------------------------------------

# How each scale degree normally functions, as (semitones above tonic,
# chord quality family, score).  A pitch class histogram cannot tell C
# major from A minor - they contain exactly the same seven notes - so a
# histogram-only key finder is guessing on every piece in a minor key.
# What separates them is which chords appear, how long they last, and
# where they sit: a piece in A minor leans on Am, uses E or E7 to pull
# back to it, and very often starts or ends there.

_MAJOR_FUNCTIONS = {
    0:  {"maj": 3.2, "maj7": 3.0, "6": 2.8, "add9": 2.8, "maj9": 2.6, "min": -0.6, "7": 1.2},
    2:  {"min": 2.0, "min7": 2.1, "min9": 1.8, "maj": 0.2, "7": 0.5, "m7b5": 0.4},
    4:  {"min": 1.7, "min7": 1.7, "maj": 0.3, "7": 0.5},
    5:  {"maj": 2.6, "maj7": 2.3, "6": 2.0, "min": 0.4, "7": 0.9, "add9": 2.0},
    7:  {"maj": 2.8, "7": 3.0, "sus4": 2.2, "9": 2.4, "13": 2.2, "min": 0.3, "min7": 0.5},
    9:  {"min": 2.2, "min7": 2.2, "min9": 1.9, "maj": 0.3, "7": 0.5},
    11: {"dim": 1.4, "m7b5": 1.6, "dim7": 1.2, "min": 0.1, "maj": -0.4},
}

_MINOR_FUNCTIONS = {
    0:  {"min": 3.2, "min7": 2.9, "min6": 2.7, "min9": 2.6, "maj": -0.3, "7": 0.3},
    2:  {"dim": 1.5, "m7b5": 1.7, "min": 0.9, "maj": 0.1},
    3:  {"maj": 2.4, "maj7": 2.2, "6": 1.8, "min": 0.2},
    5:  {"min": 2.4, "min7": 2.3, "maj": 0.9, "7": 0.8},
    7:  {"maj": 2.7, "7": 3.0, "min": 2.0, "min7": 1.9, "sus4": 1.8, "9": 2.2},
    8:  {"maj": 2.4, "maj7": 2.0, "6": 1.6, "min": 0.2},
    10: {"maj": 2.3, "7": 2.0, "maj7": 1.6, "min": 0.2},
    11: {"dim": 1.5, "dim7": 1.8, "m7b5": 1.0},
}

_QUALITY_FAMILY_FALLBACK = {
    "maj": "maj", "maj7": "maj7", "maj9": "maj9", "6": "6", "add9": "add9",
    "min": "min", "min7": "min7", "min9": "min9", "min6": "min6",
    "7": "7", "9": "9", "11": "7", "13": "13",
    "dim": "dim", "dim7": "dim7", "m7b5": "m7b5",
    "aug": "maj", "sus2": "sus4", "sus4": "sus4",
}

#: Score given to a chord that does not belong to the key at all.  It is
#: negative but small: real songs borrow chords, modulate briefly and use
#: secondary dominants, so a non-diatonic chord is evidence against a key,
#: not proof.
_NON_DIATONIC = -0.9


def _chord_function_score(root_pc: int, quality: str, tonic: int, mode: str) -> float:
    table = _MAJOR_FUNCTIONS if mode == "major" else _MINOR_FUNCTIONS
    degree = (root_pc - tonic) % 12
    entry = table.get(degree)
    if entry is None:
        return _NON_DIATONIC
    fam = _QUALITY_FAMILY_FALLBACK.get(quality, quality)
    if fam in entry:
        return entry[fam]
    # Right root, unexpected quality: still some evidence for the key,
    # since the root is diatonic.
    return 0.35


def estimate_key_from_chords(
    chord_spans: List[Tuple[float, int, str]],
    chroma_hist: Optional[np.ndarray] = None,
    chroma_weight: float = 0.30,
    edge_bonus: float = 1.6,
    cadence_bonus: float = 1.2,
) -> KeyEstimate:
    """Estimate the key from the detected chords, not just from the notes.

    ``chord_spans`` is ``[(duration_seconds, root_pc, quality), ...]`` in
    playing order.

    Three kinds of evidence are combined:

    1. **Function and duration.**  Every chord votes for each of the 24
       keys according to how well it functions there, weighted by how long
       it sounds.  A chord outside the key votes against it.
    2. **Position.**  The first and last chords carry extra weight.  Songs
       overwhelmingly begin or end on the tonic, and this is the single
       strongest signal separating a key from its relative major or minor.
    3. **Cadence.**  A dominant resolving to the tonic (V to I, or V7 to
       i) is the gesture that establishes a key, so an actual occurrence
       of it is counted directly.

    The pitch class histogram is folded in with a small weight as a
    tie-breaker, since it is the weaker evidence of the two.
    """
    if not chord_spans:
        return KeyEstimate(tonic="C", mode="major", confidence=0.0, alternatives=[])

    total_dur = sum(max(d, 0.0) for d, _r, _q in chord_spans) or 1.0

    candidates: List[Tuple[float, int, str]] = []
    for tonic in range(12):
        for mode in ("major", "minor"):
            score = 0.0
            for dur, root, quality in chord_spans:
                if root < 0:
                    continue
                score += (max(dur, 0.0) / total_dur) * _chord_function_score(
                    root, quality, tonic, mode
                )

            first_dur, first_root, first_q = chord_spans[0]
            last_dur, last_root, last_q = chord_spans[-1]
            tonic_fam = "min" if mode == "minor" else "maj"
            for root, quality, w in (
                (first_root, first_q, 0.8),
                (last_root, last_q, 1.0),
            ):
                if root < 0:
                    continue
                if root % 12 == tonic:
                    fam = _QUALITY_FAMILY_FALLBACK.get(quality, quality)
                    matches_mode = fam.startswith("min") if mode == "minor" else not fam.startswith("min")
                    score += edge_bonus * w * (1.0 if matches_mode else 0.35)

            # Dominant -> tonic motion.
            dom = (tonic + 7) % 12
            for i in range(len(chord_spans) - 1):
                _d0, r0, q0 = chord_spans[i]
                _d1, r1, q1 = chord_spans[i + 1]
                if r0 % 12 != dom or r1 % 12 != tonic:
                    continue
                f0 = _QUALITY_FAMILY_FALLBACK.get(q0, q0)
                f1 = _QUALITY_FAMILY_FALLBACK.get(q1, q1)
                dominant_like = f0 in ("maj", "7", "9", "13", "maj7", "sus4")
                tonic_like = (f1.startswith("min") if mode == "minor"
                              else not f1.startswith("min"))
                if dominant_like and tonic_like:
                    score += cadence_bonus / max(1, len(chord_spans) - 1) * 3.0

            candidates.append((score, tonic, mode))

    if chroma_hist is not None and chroma_hist.sum() > 0:
        h = chroma_hist / chroma_hist.sum()
        maj = _correlate_profile(h, KK_MAJOR)
        minr = _correlate_profile(h, KK_MINOR)
        scaled = []
        for score, tonic, mode in candidates:
            corr = maj[tonic] if mode == "major" else minr[tonic]
            scaled.append((score + chroma_weight * corr * 3.0, tonic, mode))
        candidates = scaled

    candidates.sort(reverse=True)
    best_score, best_tonic, best_mode = candidates[0]
    runner_up = candidates[1][0]

    margin = max(0.0, best_score - runner_up)
    confidence = float(np.clip(margin / 0.9, 0.0, 1.0))

    alternatives = [
        {"tonic": pc_name(t), "mode": m, "name": f"{pc_name(t)} {m}",
         "confidence": round(float(np.clip((s - runner_up + margin) / 2.5, 0.0, 1.0)), 4)}
        for s, t, m in candidates[1:4]
    ]

    return KeyEstimate(
        tonic=pc_name(best_tonic),
        mode=best_mode,
        confidence=round(confidence, 4),
        alternatives=alternatives,
    )
