"""Forced alignment: giving each word its own start and end time.

The problem
-----------
Some transcription providers time every word.  Some time only whole
phrases.  The specification is explicit about what to do in the second
case: align the words inside the phrase, and never present an estimate as
if it were a measurement.

What this does
--------------
Within a phrase whose start and end are known, it finds the word
boundaries that best explain the audio, by dynamic programming over two
pieces of evidence:

1. **Where the voice stops.**  Speech energy in the band where voices
   live dips between words and dips further between phrases.  A boundary
   placed in a dip is more likely than one placed in the middle of a held
   vowel.
2. **How long each word should take.**  A word's expected duration is
   estimated from its syllable count.  "strawberry" is not the same
   length as "a", and an aligner that splits a phrase evenly gets both
   wrong.

The DP then finds the single assignment of boundaries that minimises the
total cost of both, subject to the words staying in order and inside the
phrase.

What this is not
----------------
This is **not** phoneme-level forced alignment.  A real aligner (Montreal
Forced Aligner, Gentle, a CTC model) compares the audio against an
acoustic model of the actual phonemes in each word, and is considerably
more accurate.  This uses only energy and syllable counts, so it will
place boundaries well when words are separated by audible gaps and
poorly when they run together.

That difference is why every word this produces is tagged
``forced_alignment`` rather than ``provider_word``, why the API returns
that tag, and why the editor shows such words differently.  The honest
summary is: better than dividing the phrase evenly, worse than a real
acoustic aligner, and always labelled as an estimate.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


# --- syllable estimation ----------------------------------------------------

_LATIN_VOWELS = "aeiouyàáâäãåèéêëìíîïòóôöõùúûüøæœ"

# Hebrew letters that usually carry a vowel sound when written without
# pointing (matres lectionis), plus the final forms.
_HEBREW_LETTERS = set("אבגדהוזחטיכךלמםנןסעפףצץקרשת")
_HEBREW_MATRES = set("אהוי")


def _script_of(word: str) -> str:
    for ch in word:
        if "֐" <= ch <= "׿":
            return "hebrew"
        if "؀" <= ch <= "ۿ":
            return "arabic"
        if "Ѐ" <= ch <= "ӿ":
            return "cyrillic"
        if ch.isalpha():
            return "latin"
    return "other"


def estimate_syllables(word: str, language: Optional[str] = None) -> float:
    """Approximate syllable count, used only to weight relative durations.

    Exactness is not required and not achievable without a pronunciation
    dictionary.  What matters is that long words get proportionally more
    time than short ones.
    """
    cleaned = "".join(
        ch for ch in unicodedata.normalize("NFKD", word.lower())
        if ch.isalpha() or ch in "'’"
    )
    if not cleaned:
        return 0.6

    script = _script_of(cleaned)

    if script == "hebrew":
        # Written Hebrew leaves most vowels out, so letters are a better
        # proxy than "vowels".  Roughly one syllable per two consonants,
        # with matres lectionis nudging the count up.
        letters = [c for c in cleaned if c in _HEBREW_LETTERS]
        if not letters:
            return 0.8
        base = max(1.0, len(letters) / 2.0)
        base += 0.3 * sum(1 for c in letters if c in _HEBREW_MATRES)
        return min(base, 8.0)

    if script in ("arabic", "cyrillic", "other"):
        return max(1.0, len(cleaned) / 2.4)

    # Latin script: count vowel groups, then correct the usual cases.
    groups = re.findall(f"[{_LATIN_VOWELS}]+", cleaned)
    count = len(groups)
    if cleaned.endswith("e") and count > 1 and not cleaned.endswith(("le", "ee", "ye")):
        count -= 1          # silent terminal e
    if cleaned.endswith(("es", "ed")) and count > 1:
        count -= 0.5
    return float(max(1.0, min(count if count else 1.0, 8.0)))


def expected_durations_ms(words: Sequence[str], total_ms: int,
                          language: Optional[str] = None) -> np.ndarray:
    """Share out a phrase's duration between its words by syllable weight."""
    if not words:
        return np.zeros(0)
    syl = np.array([estimate_syllables(w, language) for w in words], dtype=np.float64)
    # A per-word constant absorbs the onset and release that every word
    # has regardless of length.
    weights = syl + 0.55
    weights = weights / weights.sum()
    return weights * float(max(total_ms, 0))


# --- speech activity --------------------------------------------------------

def speech_energy_envelope(
    samples: np.ndarray,
    sr: int,
    frame_ms: float = 10.0,
    low_hz: float = 180.0,
    high_hz: float = 4200.0,
) -> Tuple[np.ndarray, float]:
    """Energy in the band where the human voice carries, per short frame.

    Restricting to roughly 180-4200 Hz keeps the bass guitar, the kick
    drum and the cymbals from dominating the envelope, which they
    otherwise do in a full mix.  Returns ``(envelope, frames_per_second)``.
    """
    if samples.ndim != 1:
        samples = samples.mean(axis=0)
    hop = max(1, int(round(sr * frame_ms / 1000.0)))
    win = hop * 4
    n_frames = max(1, 1 + (len(samples) - win) // hop) if len(samples) >= win else 1

    nfft = 1
    while nfft < win:
        nfft <<= 1
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    band = (freqs >= low_hz) & (freqs <= high_hz)

    window = np.hanning(win)
    env = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        chunk = samples[start:start + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        spec = np.abs(np.fft.rfft(chunk * window, n=nfft))
        env[i] = float(np.sqrt(np.sum(spec[band] ** 2)))

    peak = env.max()
    if peak > 0:
        env = env / peak
    return env, sr / hop


# --- the aligner -------------------------------------------------------------

@dataclass
class AlignedWord:
    text: str
    start_ms: int
    end_ms: int
    timing_source: str


def align_words_in_span(
    words: Sequence[str],
    span_start_ms: int,
    span_end_ms: int,
    envelope: Optional[np.ndarray] = None,
    envelope_fps: Optional[float] = None,
    language: Optional[str] = None,
    duration_cost_weight: float = 1.0,
    boundary_cost_weight: float = 0.9,
    min_word_ms: int = 70,
    trim_silence: bool = True,
    silence_threshold: float = 0.12,
) -> List[AlignedWord]:
    """Place ``words`` inside ``[span_start_ms, span_end_ms]``.

    With an envelope, a dynamic program balances expected word length
    against where the voice actually pauses.  Without one, the words are
    shared out by syllable weight and tagged as merely estimated.
    """
    words = [w for w in words]
    if not words:
        return []

    total_ms = max(0, span_end_ms - span_start_ms)
    if total_ms <= 0:
        return [AlignedWord(w, span_start_ms, span_start_ms, "estimated") for w in words]

    expected = expected_durations_ms(words, total_ms, language)

    if envelope is None or envelope_fps is None or envelope.size < len(words) * 2:
        out, cursor = [], float(span_start_ms)
        for w, dur in zip(words, expected):
            start = int(round(cursor))
            cursor += float(dur)
            out.append(AlignedWord(w, start, int(round(cursor)), "estimated"))
        out[-1] = AlignedWord(out[-1].text, out[-1].start_ms, span_end_ms, "estimated")
        return out

    # Work on the slice of the envelope that covers this phrase.
    f0 = int(round(span_start_ms / 1000.0 * envelope_fps))
    f1 = int(round(span_end_ms / 1000.0 * envelope_fps))
    f0 = max(0, min(f0, envelope.size - 1))
    f1 = max(f0 + 1, min(f1, envelope.size))
    local = envelope[f0:f1]

    # A provider's phrase boundary usually includes the pause before the
    # first word and after the last one, and aligning inside that padding
    # drags the first word backwards into the silence.
    #
    # Measured on the synthetic phrase in the tests: trimming cut the
    # first word's onset error from 200 ms to 50 ms and the worst word's
    # from 200 ms to 100 ms, while leaving the mean error unchanged at
    # 62 ms - it redistributes error rather than removing it. That is
    # still worth having, because one badly placed word is more visible
    # to a reader than four slightly imprecise ones.
    lead_frames = 0
    if trim_silence and local.size > 4:
        loud = np.flatnonzero(local >= silence_threshold * max(local.max(), 1e-9))
        if loud.size >= 2:
            lead_frames = int(loud[0])
            tail_frames = int(local.size - 1 - loud[-1])
            # Never trim away more than a third of the phrase.
            cap = local.size // 3
            lead_frames = min(lead_frames, cap)
            tail_frames = min(tail_frames, cap)
            if lead_frames + tail_frames < local.size - len(words):
                local = local[lead_frames: local.size - tail_frames]
                span_start_ms = span_start_ms + int(round(lead_frames * 1000.0 / envelope_fps))
                span_end_ms = span_end_ms - int(round(tail_frames * 1000.0 / envelope_fps))
                total_ms = max(1, span_end_ms - span_start_ms)
                expected = expected_durations_ms(words, total_ms, language)

    n = local.size
    n_words = len(words)

    if n < n_words + 1:
        out, cursor = [], float(span_start_ms)
        for w, dur in zip(words, expected):
            start = int(round(cursor)); cursor += float(dur)
            out.append(AlignedWord(w, start, int(round(cursor)), "estimated"))
        return out

    ms_per_frame = 1000.0 / envelope_fps
    min_frames = max(1, int(round(min_word_ms / ms_per_frame)))

    # Cost of cutting at a frame: cheap in a dip, expensive mid-voice.
    # Smoothing first stops a single noisy frame from attracting a cut.
    smooth = np.convolve(local, np.ones(3) / 3.0, mode="same")
    cut_cost = boundary_cost_weight * smooth

    exp_frames = np.maximum(expected / ms_per_frame, float(min_frames))

    INF = 1e18
    # dp[i][t] = best cost for placing the first i words, ending at frame t.
    dp = np.full((n_words + 1, n + 1), INF, dtype=np.float64)
    back = np.zeros((n_words + 1, n + 1), dtype=np.int32)
    dp[0, 0] = 0.0

    for i in range(1, n_words + 1):
        want = exp_frames[i - 1]
        # A word may run between a third and three times its expected
        # length; outside that the estimate is not worth trusting.
        lo_len = max(min_frames, int(np.floor(want / 3.0)))
        hi_len = max(lo_len + 1, int(np.ceil(want * 3.0)))

        for t in range(i, n + 1):
            start_lo = max(i - 1, t - hi_len)
            start_hi = min(t - lo_len, t - 1)
            if start_hi < start_lo:
                start_hi = max(i - 1, t - 1)
                start_lo = start_hi
            best, best_s = INF, start_lo
            for s in range(start_lo, start_hi + 1):
                prev = dp[i - 1, s]
                if prev >= INF:
                    continue
                length = t - s
                # Log-ratio keeps the penalty symmetric: half as long and
                # twice as long cost the same.
                ratio = np.log(max(length, 1) / max(want, 1e-6))
                cost = prev + duration_cost_weight * ratio * ratio
                if t < n:
                    cost += cut_cost[min(t, n - 1)]
                if cost < best:
                    best, best_s = cost, s
            dp[i, t] = best
            back[i, t] = best_s

    if dp[n_words, n] >= INF:
        out, cursor = [], float(span_start_ms)
        for w, dur in zip(words, expected):
            start = int(round(cursor)); cursor += float(dur)
            out.append(AlignedWord(w, start, int(round(cursor)), "estimated"))
        return out

    cuts = [n]
    t = n
    for i in range(n_words, 0, -1):
        t = int(back[i, t])
        cuts.append(t)
    cuts.reverse()

    out: List[AlignedWord] = []
    for i, w in enumerate(words):
        s_ms = span_start_ms + int(round(cuts[i] * ms_per_frame))
        e_ms = span_start_ms + int(round(cuts[i + 1] * ms_per_frame))
        out.append(AlignedWord(w, s_ms, max(e_ms, s_ms + 1), "forced_alignment"))
    out[-1] = AlignedWord(out[-1].text, out[-1].start_ms, span_end_ms, "forced_alignment")
    return out


def tokenize(text: str) -> List[str]:
    """Split a phrase into words, keeping attached punctuation with the word."""
    return [t for t in re.split(r"\s+", text.strip()) if t]
