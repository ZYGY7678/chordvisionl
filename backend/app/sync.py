"""Placing chords above the right words.

This is the feature the whole product is built around, so it is worth
being precise about what "above the right word" means.

Both chords and words carry start and end times in milliseconds on the
same timeline - the audio's own.  A chord belongs above the word whose
span contains the moment the chord starts.  That is the entire rule, and
it is deliberately the entire rule: the alternative approach, dividing
the chords evenly across the words of a line, produces something that
looks plausible and is wrong everywhere the music is not perfectly
regular.  The specification rules it out and so does this module.

Three cases need care:

**A chord change inside a word.**  Held notes get chord changes under
them all the time.  The chord is anchored to that word with an offset
expressing how far into the word it lands, so the renderer can place the
symbol over the right syllable instead of over the start of the word.

**A chord change between words.**  Common in gaps and at line ends.  The
chord is anchored to the following word as a leading marker, so it
appears before the word rather than on top of it.

**A chord that starts before any lyrics.**  Intros, instrumental breaks
and solos.  These become their own instrumental lines rather than being
forced onto the first sung word.

Nothing here moves a chord in time.  The original ``start_time_ms`` is
preserved on every placement, because that is what the player
synchronises to and what the editor edits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Dict, Any


@dataclass
class PlacedChord:
    """A chord positioned relative to a word."""
    chord_id: str
    chord_name: str
    start_time_ms: int
    end_time_ms: int
    confidence: float
    word_index: Optional[int]
    """Index into the line's words, or None for an instrumental line."""
    offset_ratio: float = 0.0
    """How far into that word the chord starts, 0 at its beginning and 1
    at its end.  The renderer uses this to position the symbol over the
    correct syllable."""
    anchor: str = "on_word"
    """``on_word``, ``before_word`` (a gap) or ``instrumental``."""
    is_edited: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chordId": self.chord_id, "chordName": self.chord_name,
            "startTimeMs": self.start_time_ms, "endTimeMs": self.end_time_ms,
            "confidence": round(self.confidence, 4),
            "wordIndex": self.word_index,
            "offsetRatio": round(self.offset_ratio, 4),
            "anchor": self.anchor, "isEdited": self.is_edited,
        }


@dataclass
class LyricLine:
    line_id: int
    words: List[Dict[str, Any]] = field(default_factory=list)
    chords: List[PlacedChord] = field(default_factory=list)
    start_time_ms: int = 0
    end_time_ms: int = 0
    is_instrumental: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lineId": self.line_id,
            "startTimeMs": self.start_time_ms,
            "endTimeMs": self.end_time_ms,
            "isInstrumental": self.is_instrumental,
            "words": self.words,
            "chords": [c.to_dict() for c in self.chords],
        }


def group_words_into_lines(
    words: Sequence[Dict[str, Any]],
    max_gap_ms: int = 900,
    max_words_per_line: int = 11,
    max_line_ms: int = 9000,
) -> List[List[Dict[str, Any]]]:
    """Break a flat word list into lines, the way a lyric sheet reads.

    A line ends at a long pause, at sentence punctuation, or when it has
    simply grown too long to read comfortably.  Providers that return
    their own segments should be preferred over this; it is the fallback
    for the ones that do not.
    """
    lines: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for i, w in enumerate(words):
        if current:
            gap = int(w["startTimeMs"]) - int(current[-1]["endTimeMs"])
            span = int(w["endTimeMs"]) - int(current[0]["startTimeMs"])
            ends_sentence = str(current[-1].get("text", "")).endswith((".", "?", "!", "…", ":", ";"))
            if (gap > max_gap_ms or len(current) >= max_words_per_line
                    or span > max_line_ms or (ends_sentence and len(current) >= 3)):
                lines.append(current)
                current = []
        current.append(dict(w))

    if current:
        lines.append(current)
    return lines


def synchronize(
    words: Sequence[Dict[str, Any]],
    chords: Sequence[Dict[str, Any]],
    instrumental_gap_ms: int = 2500,
) -> List[LyricLine]:
    """Build the chord-over-lyric view.

    ``words`` and ``chords`` are dicts with ``startTimeMs`` / ``endTimeMs``
    in the same timeline.  Returns the lines, each carrying its words and
    the chords placed against them.
    """
    real_chords = [c for c in chords if c.get("chordName") not in ("N", None, "")]
    sorted_words = sorted(words, key=lambda w: int(w["startTimeMs"]))
    sorted_chords = sorted(real_chords, key=lambda c: int(c["startTimeMs"]))

    if not sorted_words:
        return [_instrumental_line(0, sorted_chords)] if sorted_chords else []

    # If the words already carry a line id - set when the transcript was
    # stored, from the provider's own phrase boundaries - respect it.
    # Re-deriving lines here would silently override better information.
    if any("lineId" in w for w in sorted_words):
        buckets: Dict[int, List[Dict[str, Any]]] = {}
        for w in sorted_words:
            buckets.setdefault(int(w.get("lineId", 0)), []).append(w)
        raw_lines = [buckets[k] for k in sorted(buckets)]
    else:
        raw_lines = group_words_into_lines(sorted_words)

    lines: List[LyricLine] = []
    line_id = 0

    # Every chord is placed exactly once. Without this, an intro chord
    # ends up both on its own instrumental line and as a leading marker
    # before the first sung word, and the reader sees it twice.
    consumed: set = set()

    # An instrumental intro: chords before the first word.
    first_word_start = int(sorted_words[0]["startTimeMs"])
    intro = [c for c in sorted_chords if int(c["endTimeMs"]) <= first_word_start]
    if intro and first_word_start >= instrumental_gap_ms:
        lines.append(_instrumental_line(line_id, intro))
        line_id += 1
        consumed.update(id(c) for c in intro)

    for idx, line_words in enumerate(raw_lines):
        line_start = int(line_words[0]["startTimeMs"])
        line_end = int(line_words[-1]["endTimeMs"])

        # Chords that belong to this line: those starting between this
        # line's first word and the next line's first word.
        next_start = (int(raw_lines[idx + 1][0]["startTimeMs"])
                      if idx + 1 < len(raw_lines) else None)

        # Where this line's claim on the chords stops.
        #
        # Normally it runs up to the next line's first word. But when
        # there is a long instrumental gap after this line, a chord that
        # starts inside that gap belongs to the break, not stapled onto
        # the line's last word - a solo is not part of the previous
        # lyric. Without this the chord lands on the final word with an
        # offset of 1.0, which reads as if it were sung there.
        if next_start is not None and (next_start - line_end) >= instrumental_gap_ms:
            upper = line_end
        elif next_start is not None:
            upper = next_start
        else:
            upper = 10 ** 12
        lower = line_start if idx > 0 else 0

        line_chords = [c for c in sorted_chords
                       if lower <= int(c["startTimeMs"]) < upper
                       and id(c) not in consumed]

        line = LyricLine(
            line_id=line_id,
            words=[dict(w) for w in line_words],
            start_time_ms=line_start,
            end_time_ms=line_end,
        )

        for c in line_chords:
            line.chords.append(_place_chord(c, line_words))
            consumed.add(id(c))
        lines.append(line)
        line_id += 1

        # An instrumental break between this line and the next.
        if next_start is not None and (next_start - line_end) >= instrumental_gap_ms:
            between = [c for c in sorted_chords
                       if line_end <= int(c["startTimeMs"]) < next_start
                       and id(c) not in consumed]
            if between:
                lines.append(_instrumental_line(line_id, between))
                consumed.update(id(c) for c in between)
                line_id += 1

    return lines


def _place_chord(chord: Dict[str, Any],
                 line_words: Sequence[Dict[str, Any]]) -> PlacedChord:
    """Anchor one chord to one word of a line."""
    start = int(chord["startTimeMs"])

    target_index: Optional[int] = None
    anchor = "before_word"
    offset = 0.0

    for i, w in enumerate(line_words):
        ws, we = int(w["startTimeMs"]), int(w["endTimeMs"])
        if ws <= start < we:
            target_index = i
            anchor = "on_word"
            span = max(1, we - ws)
            offset = (start - ws) / span
            break
        if start < ws:
            # The chord changes in the gap before this word.
            target_index = i
            anchor = "before_word"
            offset = 0.0
            break

    if target_index is None:
        # After the last word of the line: attach to it, at its end.
        target_index = len(line_words) - 1
        anchor = "on_word"
        offset = 1.0

    return PlacedChord(
        chord_id=str(chord.get("id", "")),
        chord_name=str(chord["chordName"]),
        start_time_ms=start,
        end_time_ms=int(chord["endTimeMs"]),
        confidence=float(chord.get("confidence", 0.0)),
        word_index=target_index,
        offset_ratio=float(min(max(offset, 0.0), 1.0)),
        anchor=anchor,
        is_edited=bool(chord.get("isEdited", False)),
    )


def _instrumental_line(line_id: int, chords: Sequence[Dict[str, Any]]) -> LyricLine:
    placed = [
        PlacedChord(
            chord_id=str(c.get("id", "")),
            chord_name=str(c["chordName"]),
            start_time_ms=int(c["startTimeMs"]),
            end_time_ms=int(c["endTimeMs"]),
            confidence=float(c.get("confidence", 0.0)),
            word_index=None, anchor="instrumental",
            is_edited=bool(c.get("isEdited", False)),
        )
        for c in chords
    ]
    return LyricLine(
        line_id=line_id, words=[], chords=placed,
        start_time_ms=min((p.start_time_ms for p in placed), default=0),
        end_time_ms=max((p.end_time_ms for p in placed), default=0),
        is_instrumental=True,
    )
