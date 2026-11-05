"""Measurement harness for the chord engine.

The metrics here are the ones the music information retrieval field uses
(MIREX), not metrics invented to make the engine look good.

Chord Symbol Recall (CSR)
    The share of the audio's *duration* for which the estimated chord
    matches the reference.  Duration-weighted, not segment-counted,
    because getting a four-bar chord right matters more than getting a
    passing chord right.

Vocabulary levels
    The same output is scored at several levels of detail, because "wrong"
    is not one thing:
      * ``root``     - is the root right, ignoring quality entirely
      * ``majmin``   - root plus major/minor, the standard coarse measure
      * ``triad``    - root plus the triad type (maj/min/dim/aug/sus)
      * ``sevenths`` - the full quality, sevenths and all
      * ``bass``     - full quality *and* the inversion
    Scores necessarily fall as the vocabulary gets stricter.  Quoting only
    the loosest level is the usual way these systems are oversold.

Segmentation
    Chord *identity* and chord *timing* are separate skills and are
    measured separately.  Boundary F1 at a tolerance counts how many
    reference chord changes have a detected change nearby, and how many
    detected changes are spurious.
"""
from __future__ import annotations

import sys, os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Sequence, Callable

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine"))

from chordvision_engine.theory import parse_chord, NO_CHORD, CHORD_QUALITIES


# --- vocabulary mappings ---------------------------------------------------

_MAJ_FAMILY = {"maj", "maj7", "7", "6", "add9", "9", "maj9", "11", "13", "aug"}
_MIN_FAMILY = {"min", "min7", "min6", "min9", "dim", "dim7", "m7b5"}
_SUS_FAMILY = {"sus2", "sus4"}

_TRIAD_OF = {
    "maj": "maj", "maj7": "maj", "7": "maj", "6": "maj", "add9": "maj",
    "9": "maj", "maj9": "maj", "13": "maj", "aug": "aug",
    "min": "min", "min7": "min", "min6": "min", "min9": "min",
    "dim": "dim", "dim7": "dim", "m7b5": "dim",
    "sus2": "sus2", "sus4": "sus4", "11": "sus4",
}


def _reduce(root: int, quality: str, bass: Optional[int], level: str):
    """Reduce a chord to the comparison key for a given vocabulary level."""
    if quality == NO_CHORD:
        return ("N",)
    if level == "root":
        return (root % 12,)
    if level == "majmin":
        if quality in _MAJ_FAMILY:
            fam = "maj"
        elif quality in _MIN_FAMILY:
            fam = "min"
        elif quality in _SUS_FAMILY:
            fam = "sus"
        else:
            fam = quality
        return (root % 12, fam)
    if level == "triad":
        return (root % 12, _TRIAD_OF.get(quality, quality))
    if level == "sevenths":
        return (root % 12, quality)
    if level == "bass":
        b = bass % 12 if bass is not None else root % 12
        return (root % 12, quality, b)
    raise ValueError(f"unknown vocabulary level: {level}")


VOCAB_LEVELS = ("root", "majmin", "triad", "sevenths", "bass")


# --- interval arithmetic ---------------------------------------------------

@dataclass
class Interval:
    start_s: float
    end_s: float
    root: int
    quality: str
    bass: Optional[int]

    @staticmethod
    def from_symbol(symbol: str, start_s: float, end_s: float) -> "Interval":
        r, q, b = parse_chord(symbol)
        return Interval(start_s, end_s, r, q, b)


def _overlap(a: Interval, b: Interval) -> float:
    return max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))


def chord_symbol_recall(
    reference: Sequence[Interval],
    estimate: Sequence[Interval],
    level: str,
    ignore_no_chord: bool = True,
) -> Tuple[float, float]:
    """Duration-weighted recall at one vocabulary level.

    Returns ``(recall, scored_duration_s)``.  Reference spans marked "no
    chord" are excluded by default: a song's silent intro should neither
    reward nor punish the engine.
    """
    correct = 0.0
    total = 0.0
    for ref in reference:
        if ignore_no_chord and ref.quality == NO_CHORD:
            continue
        dur = ref.end_s - ref.start_s
        if dur <= 0:
            continue
        total += dur
        ref_key = _reduce(ref.root, ref.quality, ref.bass, level)
        for est in estimate:
            ov = _overlap(ref, est)
            if ov <= 0:
                continue
            est_key = _reduce(est.root, est.quality, est.bass, level)
            if est_key == ref_key:
                correct += ov
    if total <= 0:
        return (0.0, 0.0)
    return (correct / total, total)


def boundary_scores(
    reference: Sequence[Interval],
    estimate: Sequence[Interval],
    tolerance_s: float = 0.25,
) -> Dict[str, float]:
    """Precision / recall / F1 over chord change times."""
    def boundaries(spans: Sequence[Interval]) -> np.ndarray:
        if not spans:
            return np.zeros(0)
        b = [s.start_s for s in spans[1:]]
        return np.asarray(sorted(b))

    ref_b = boundaries(list(reference))
    est_b = boundaries(list(estimate))

    if ref_b.size == 0 and est_b.size == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                "median_abs_error_s": 0.0, "n_ref": 0, "n_est": 0}
    if ref_b.size == 0 or est_b.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "median_abs_error_s": float("nan"),
                "n_ref": int(ref_b.size), "n_est": int(est_b.size)}

    # Greedy one-to-one matching, nearest first.
    pairs = []
    for i, r in enumerate(ref_b):
        for j, e in enumerate(est_b):
            d = abs(r - e)
            if d <= tolerance_s:
                pairs.append((d, i, j))
    pairs.sort()

    used_r, used_e = set(), set()
    errors = []
    for d, i, j in pairs:
        if i in used_r or j in used_e:
            continue
        used_r.add(i)
        used_e.add(j)
        errors.append(d)

    tp = len(used_r)
    precision = tp / len(est_b)
    recall = tp / len(ref_b)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_abs_error_s": float(np.median(errors)) if errors else float("nan"),
        "n_ref": int(ref_b.size),
        "n_est": int(est_b.size),
    }


def key_score(ref_tonic_pc: int, ref_mode: str, est_tonic_pc: int, est_mode: str) -> float:
    """MIREX-weighted key score.

    Exact match scores 1.  Near misses score partial credit because they
    are musically close and a human would call them understandable:
    a fifth away 0.5, the relative major/minor 0.3, the parallel
    major/minor 0.2.  Everything else scores 0.
    """
    if ref_tonic_pc == est_tonic_pc and ref_mode == est_mode:
        return 1.0
    if ref_mode == est_mode:
        d = (est_tonic_pc - ref_tonic_pc) % 12
        if d in (7, 5):
            return 0.5
    if ref_mode == "major" and est_mode == "minor":
        if (est_tonic_pc - ref_tonic_pc) % 12 == 9:
            return 0.3
        if est_tonic_pc == ref_tonic_pc:
            return 0.2
    if ref_mode == "minor" and est_mode == "major":
        if (est_tonic_pc - ref_tonic_pc) % 12 == 3:
            return 0.3
        if est_tonic_pc == ref_tonic_pc:
            return 0.2
    return 0.0


@dataclass
class ItemResult:
    name: str
    csr: Dict[str, float]
    boundaries: Dict[str, float]
    key_points: float
    key_ref: str
    key_est: str
    duration_s: float
    analysis_s: float
    tuning_cents_ref: float
    tuning_cents_est: float


def aggregate(results: Sequence[ItemResult]) -> Dict[str, object]:
    """Duration-weighted aggregation across the whole test set."""
    if not results:
        return {}
    durs = np.array([r.duration_s for r in results], dtype=float)
    w = durs / durs.sum()

    out: Dict[str, object] = {"n_items": len(results), "total_duration_s": float(durs.sum())}
    for level in VOCAB_LEVELS:
        vals = np.array([r.csr.get(level, 0.0) for r in results])
        out[f"csr_{level}"] = float(np.sum(vals * w))
        out[f"csr_{level}_worst"] = float(vals.min())

    for k in ("precision", "recall", "f1"):
        vals = np.array([r.boundaries.get(k, 0.0) for r in results])
        out[f"boundary_{k}"] = float(np.sum(vals * w))

    errs = np.array([r.boundaries.get("median_abs_error_s", np.nan) for r in results])
    errs = errs[~np.isnan(errs)]
    out["boundary_median_error_ms"] = float(np.median(errs) * 1000) if errs.size else None

    out["key_score"] = float(np.mean([r.key_points for r in results]))
    out["key_exact"] = float(np.mean([1.0 if r.key_points >= 1.0 else 0.0 for r in results]))

    tune_err = np.array([abs(r.tuning_cents_est - r.tuning_cents_ref) for r in results])
    out["tuning_mean_abs_error_cents"] = float(tune_err.mean())
    out["tuning_max_abs_error_cents"] = float(tune_err.max())

    total_audio = float(durs.sum())
    total_cpu = float(sum(r.analysis_s for r in results))
    out["realtime_factor"] = round(total_audio / total_cpu, 1) if total_cpu > 0 else None
    return out
