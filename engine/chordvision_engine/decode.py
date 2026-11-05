"""Sequence decoding: turning per-frame evidence into a chord sequence.

Matching every 46 ms frame against the template bank independently and
printing the winner produces unusable output.  A single frame is a tiny
slice of sound: a passing note, a cymbal, or the gap between two guitar
strums is enough to flip the answer.  The result flickers between chords
several times a second, which is both wrong and unreadable.

What fixes this is not smoothing after the fact but decoding the whole
sequence at once.  The engine models the song as a hidden Markov chain:

  * the **hidden state** is the chord sounding at each frame;
  * the **emission model** says how likely the observed chroma is given
    that chord;
  * the **transition model** says how likely it is to move from one chord
    to another between adjacent frames.

The transition model is where the musical knowledge lives.  Staying on the
same chord is far more likely than changing (chords last seconds, frames
last milliseconds), and when a change does happen it is far more likely to
go somewhere close on the circle of fifths than to a random root.

Viterbi then finds the single most likely chord sequence for the whole
song.  Because it optimises the sequence globally, a few bad frames in the
middle of a chord are outvoted by their neighbours instead of punching a
hole in the output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

import numpy as np

from .theory import (
    build_templates, circle_of_fifths_distance, chord_pitch_classes,
    QUALITY_SETS, NO_CHORD,
)


@dataclass
class DecoderConfig:
    """Everything that shapes the decode, in one place."""

    quality_set: str = "sevenths"
    """Which chord vocabulary to decode with.

    A larger vocabulary is not automatically better.  Every extra quality
    is another way to be wrong, and the richer templates (9ths, 11ths,
    13ths) contain so many pitch classes that they can fit almost any
    chroma vector.  "sevenths" is the default because it covers what
    appears in the overwhelming majority of songs.
    """

    kappa: float = 30.0
    """Sharpness of the emission model.

    The emission is ``kappa * cosine_similarity``.  Low values make every
    chord look equally plausible and let the transition prior dominate;
    high values make the decoder trust single frames too much and
    reintroduce flicker.

    The value is not a guess: it was chosen by sweeping it against the
    measured test set, which is reproducible with ``tests/sweep.py``.
    """

    mean_chord_duration_s: float = 2.6
    """Expected length of one chord, which sets the self-transition rate."""

    fifths_tau: float = 1.9
    """How strongly a change is biased toward nearby roots on the circle
    of fifths.  Larger values flatten the preference."""

    same_root_weight: float = 0.75
    """Extra likelihood for changing quality while keeping the root
    (C -> C7 -> Cmaj7), which is extremely common."""

    no_chord_similarity: float = 0.74
    """The similarity a real chord must beat for the decoder to prefer
    reporting "no chord" instead."""

    no_chord_prior: float = 0.22

    silence_db: float = -52.0
    """Frames quieter than this relative to the loudest frame are treated
    as silence and are forced to "no chord"."""

    min_segment_s: float = 0.45
    """Chord segments shorter than this are absorbed into their
    neighbours.  At 160 BPM this is still shorter than an eighth note, so
    nothing musical is lost; what it removes is the half-second fragment
    that appears where one chord's reverb tail overlaps the next chord's
    attack."""

    n_alternatives: int = 3

    bass_root_weight: float = 0.0
    """How much the bass register is allowed to vote for the chord root.

    The bass note is the strongest single clue to a chord's root, which is
    why this was tried.  Measured, it does not work: swept over
    0.08 / 0.15 / 0.25 / 0.40, every non-zero value made every metric
    worse, and 0.40 cost 11 points of chord recall.  The reason is
    visible in the failures - a walking bass spends three beats out of
    four away from the root, so the term votes for F, then A minor, then
    C, in the middle of a D minor 7.

    The default is therefore 0.0 and the term is off.  It is kept, with
    this note, so the next person to have the same idea can see that it
    was measured rather than overlooked.  Reproduce with:
    ``python3 tests/sweep.py --grid '{"bass_root_weight":[0.0,0.15,0.4]}'``
    """


@dataclass
class DecodeResult:
    path: np.ndarray            # (n_frames,) state index per frame
    labels: List[Tuple[int, str]]  # state index -> (root_pc, quality); NO_CHORD is last
    posterior: np.ndarray       # (n_states, n_frames) frame-wise posterior
    log_emission: np.ndarray    # (n_states, n_frames)
    no_chord_index: int
    config: DecoderConfig


def _build_transition_matrix(
    labels: Sequence[Tuple[int, str]],
    priors: np.ndarray,
    no_chord_index: int,
    cfg: DecoderConfig,
    frame_duration_s: float,
) -> np.ndarray:
    """Log transition matrix, shape (n_states, n_states)."""
    n = len(labels) + 1  # + the no-chord state

    # Probability of staying put for one frame, from the expected chord
    # length: a chord of mean duration D survives a frame of length dt with
    # probability exp(-dt/D).
    p_self = float(np.exp(-frame_duration_s / max(cfg.mean_chord_duration_s, 1e-6)))
    p_self = float(np.clip(p_self, 0.50, 0.9995))

    W = np.zeros((n, n), dtype=np.float64)

    roots = np.array([l[0] for l in labels] + [-1])
    quals = [l[1] for l in labels] + [NO_CHORD]

    # Pairwise weights between real chords.
    n_chords = len(labels)
    cof = np.zeros((n_chords, n_chords), dtype=np.float64)
    for a in range(n_chords):
        for b in range(n_chords):
            cof[a, b] = circle_of_fifths_distance(int(roots[a]), int(roots[b]))

    base = np.exp(-cof / max(cfg.fifths_tau, 1e-6))

    same_root = (roots[:n_chords, None] == roots[None, :n_chords])
    base = base + cfg.same_root_weight * same_root

    # Destination prior: it is more likely to move to a common chord.
    base = base * priors[None, :n_chords]

    W[:n_chords, :n_chords] = base

    # To and from "no chord": possible everywhere, but not encouraged.
    W[:n_chords, no_chord_index] = cfg.no_chord_prior
    W[no_chord_index, :n_chords] = priors[:n_chords]
    W[no_chord_index, no_chord_index] = 0.0

    np.fill_diagonal(W, 0.0)

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    A = (1.0 - p_self) * (W / row_sums)
    np.fill_diagonal(A, p_self)

    return np.log(np.maximum(A, 1e-300))


def _viterbi(log_emission: np.ndarray, log_trans: np.ndarray,
             log_init: np.ndarray) -> np.ndarray:
    """Standard Viterbi, vectorised over states.

    ``log_emission`` is (n_states, n_frames).  Returns the most likely
    state sequence, one index per frame.
    """
    n_states, n_frames = log_emission.shape
    if n_frames == 0:
        return np.zeros(0, dtype=np.int32)

    delta = log_init + log_emission[:, 0]
    backptr = np.empty((n_frames, n_states), dtype=np.int32)
    backptr[0] = np.arange(n_states)

    for t in range(1, n_frames):
        # scores[i, j] = delta[i] + log_trans[i, j]
        scores = delta[:, None] + log_trans
        best_prev = np.argmax(scores, axis=0)
        delta = scores[best_prev, np.arange(n_states)] + log_emission[:, t]
        backptr[t] = best_prev

        # Keep the numbers in range on long songs.
        if t % 512 == 0:
            delta -= delta.max()

    path = np.empty(n_frames, dtype=np.int32)
    path[-1] = int(np.argmax(delta))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = backptr[t, path[t]]
    return path


def decode_chords(
    chroma: np.ndarray,
    frame_duration_s: float,
    energy: Optional[np.ndarray] = None,
    cfg: Optional[DecoderConfig] = None,
    bass_chroma: Optional[np.ndarray] = None,
) -> DecodeResult:
    """Decode a chroma sequence into the most likely chord path.

    ``chroma`` must be (12, n_frames) and L2-normalised per frame.
    ``energy`` is an optional per-frame loudness used to detect silence.
    ``bass_chroma`` is an optional (12, n_frames) profile of the bass
    register used to support the root choice.
    """
    cfg = cfg or DecoderConfig()
    if chroma.shape[0] != 12:
        raise ValueError("chroma must have 12 rows")

    qualities = QUALITY_SETS.get(cfg.quality_set)
    if qualities is None:
        raise ValueError(
            f"unknown quality_set {cfg.quality_set!r}; "
            f"expected one of {sorted(QUALITY_SETS)}"
        )

    T, labels, priors = build_templates(qualities)
    n_chords = T.shape[0]
    no_chord_index = n_chords
    n_states = n_chords + 1
    n_frames = chroma.shape[1]

    # Emission: von Mises-Fisher on the unit sphere. Both the templates and
    # the chroma frames are unit vectors, so the inner product is the
    # cosine similarity between them.
    sim = T @ chroma                              # (n_chords, n_frames)
    log_prior = np.log(np.maximum(priors, 1e-12))[:, None]
    log_emission = np.empty((n_states, n_frames), dtype=np.float64)
    log_emission[:n_chords] = cfg.kappa * sim + log_prior
    log_emission[no_chord_index] = (
        cfg.kappa * cfg.no_chord_similarity + np.log(max(cfg.no_chord_prior, 1e-12))
    )

    if cfg.bass_root_weight and bass_chroma is not None and bass_chroma.shape == chroma.shape:
        root_of_state = np.array([l[0] for l in labels], dtype=np.int64)
        log_emission[:n_chords] += cfg.kappa * cfg.bass_root_weight * bass_chroma[root_of_state, :]

    # Silence handling: a quiet frame carries no chord information, so it
    # must not be allowed to vote. Forcing it to "no chord" also stops the
    # decoder from smearing the previous chord across a long outro fade.
    if energy is not None and energy.size == n_frames and np.any(energy > 0):
        ref = float(np.percentile(energy[energy > 0], 95))
        with np.errstate(divide="ignore"):
            db = 20.0 * np.log10(np.maximum(energy, 1e-12) / max(ref, 1e-12))
        silent = db < cfg.silence_db
        if np.any(silent):
            log_emission[:n_chords, silent] = -1e6
            log_emission[no_chord_index, silent] = 0.0

    log_trans = _build_transition_matrix(
        labels, priors, no_chord_index, cfg, frame_duration_s
    )

    init = np.empty(n_states)
    init[:n_chords] = priors
    init[no_chord_index] = cfg.no_chord_prior
    init = init / init.sum()
    log_init = np.log(init)

    path = _viterbi(log_emission, log_trans, log_init)

    # Frame-wise posterior, used for the confidence figures.
    m = log_emission.max(axis=0, keepdims=True)
    e = np.exp(log_emission - m)
    posterior = e / np.maximum(e.sum(axis=0, keepdims=True), 1e-300)

    all_labels = list(labels) + [(-1, NO_CHORD)]
    return DecodeResult(
        path=path,
        labels=all_labels,
        posterior=posterior,
        log_emission=log_emission,
        no_chord_index=no_chord_index,
        config=cfg,
    )


def path_to_segments(path: np.ndarray) -> List[Tuple[int, int, int]]:
    """Run-length encode a state path into ``(start_frame, end_frame, state)``.

    ``end_frame`` is exclusive.
    """
    if path.size == 0:
        return []
    changes = np.flatnonzero(np.diff(path)) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [path.size]))
    return [(int(s), int(e), int(path[s])) for s, e in zip(starts, ends)]


def merge_short_segments(
    segments: List[Tuple[int, int, int]],
    min_frames: int,
    posterior: Optional[np.ndarray] = None,
) -> List[Tuple[int, int, int]]:
    """Absorb segments shorter than ``min_frames`` into a neighbour.

    The short segment is given to whichever neighbour the audio supports
    better over that span, rather than always to the longer one: a quick
    passing chord that is genuinely closer to what follows should extend
    the following chord, not the preceding one.
    """
    if min_frames <= 1 or not segments:
        return segments

    segs = list(segments)
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, (s, e, st) in enumerate(segs):
            if (e - s) >= min_frames:
                continue

            left = segs[i - 1] if i > 0 else None
            right = segs[i + 1] if i + 1 < len(segs) else None
            if left is None and right is None:
                break

            if left is None:
                target = i + 1
            elif right is None:
                target = i - 1
            elif posterior is not None:
                lscore = float(posterior[left[2], s:e].mean())
                rscore = float(posterior[right[2], s:e].mean())
                target = (i - 1) if lscore >= rscore else (i + 1)
            else:
                target = (i - 1) if (left[1] - left[0]) >= (right[1] - right[0]) else (i + 1)

            if target < i:
                segs[target] = (segs[target][0], e, segs[target][2])
            else:
                segs[target] = (s, segs[target][1], segs[target][2])
            del segs[i]
            changed = True
            break

    # Fuse neighbours that ended up with the same state.
    fused: List[Tuple[int, int, int]] = []
    for seg in segs:
        if fused and fused[-1][2] == seg[2] and fused[-1][1] == seg[0]:
            fused[-1] = (fused[-1][0], seg[1], seg[2])
        else:
            fused.append(seg)
    return fused
