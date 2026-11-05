"""Exporting a finished song: ChordPro, text, Markdown, JSON, PDF, HTML.

The difficult part of all of this is Hebrew, and it is worth saying why.

A chord sheet works by vertical alignment: the chord symbol has to sit
directly above the syllable where the chord changes.  In English that is
simple, because the text runs left to right and so does the measuring.
Hebrew runs right to left, chord symbols are Latin and run left to right,
and the two are mixed on the same line.

So the layout is done word by word rather than by formatting a string:
each word is measured, placed at a computed x position, and the chord is
placed above the point inside that word where the chord actually starts.
For a right-to-left line the words are laid out from the right margin
leftwards, and the chord's offset inside a word is measured from the
word's right edge.  Getting this wrong puts every chord above the wrong
word, which is the one failure that would make the export useless.

Plain text cannot do any of this.  A monospaced chord line above a
right-to-left lyric line is misaligned in every editor, because the
editor re-orders the lyric for display and leaves the chord line alone.
The text export therefore switches to inline ``[Am]`` notation for
right-to-left lyrics, which is unambiguous, and keeps the familiar
chord-above-lyric columns for left-to-right ones.
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

RTL_RANGES = (
    ("\u0590", "\u05FF"),   # Hebrew
    ("\u0600", "\u06FF"),   # Arabic
    ("\u0700", "\u074F"),   # Syriac
    ("\uFB1D", "\uFDFF"),   # Hebrew / Arabic presentation forms
    ("\uFE70", "\uFEFF"),
)


def is_rtl_char(ch: str) -> bool:
    return any(lo <= ch <= hi for lo, hi in RTL_RANGES)


def line_direction(text: str) -> str:
    """"rtl" or "ltr", decided by which script carries the line."""
    rtl = sum(1 for ch in text if is_rtl_char(ch))
    ltr = sum(1 for ch in text if ch.isalpha() and not is_rtl_char(ch))
    return "rtl" if rtl > ltr else "ltr"


def _shape(text: str) -> str:
    """Reorder one word for visual rendering in a PDF.

    reportlab draws glyphs in the order it is given them, so a Hebrew
    word has to be reversed before being drawn.  python-bidi implements
    the Unicode bidirectional algorithm; if it is unavailable we fall
    back to reversing the run, which is correct for a plain Hebrew word
    and wrong for mixed content - so the fallback is a degradation, not a
    silent equivalent.
    """
    try:
        from bidi.algorithm import get_display
        return get_display(text)
    except Exception:
        if any(is_rtl_char(c) for c in text):
            return text[::-1]
        return text


# --- document assembly ------------------------------------------------------

@dataclass
class ExportDocument:
    """Everything an exporter needs, already transposed and ordered."""
    title: str
    key_name: Optional[str]
    tempo_bpm: Optional[float]
    transpose_semitones: int
    lines: List[Dict[str, Any]]
    chords_only: List[Dict[str, Any]]
    generated_note: str = ""


def build_document(
    title: str,
    lines: List[Dict[str, Any]],
    key_name: Optional[str] = None,
    tempo_bpm: Optional[float] = None,
    transpose_semitones: int = 0,
) -> ExportDocument:
    chords_only = []
    for line in lines:
        for c in line.get("chords", []):
            chords_only.append(c)
    return ExportDocument(
        title=title, key_name=key_name, tempo_bpm=tempo_bpm,
        transpose_semitones=transpose_semitones,
        lines=lines, chords_only=chords_only,
    )


# --- ChordPro ---------------------------------------------------------------

def to_chordpro(doc: ExportDocument) -> str:
    """ChordPro: chords inline in brackets, directives in braces.

    This is the format that survives everything - it is unambiguous for
    right-to-left text, and every chord app reads it.
    """
    out: List[str] = [f"{{title: {doc.title}}}"]
    if doc.key_name:
        out.append(f"{{key: {doc.key_name}}}")
    if doc.tempo_bpm:
        out.append(f"{{tempo: {doc.tempo_bpm:.0f}}}")
    if doc.transpose_semitones:
        out.append(f"{{comment: transposed {doc.transpose_semitones:+d} semitones}}")
    out.append("")

    for line in doc.lines:
        if line.get("isInstrumental"):
            symbols = " ".join(f"[{c['chordName']}]" for c in line.get("chords", []))
            out.append(f"{{comment: instrumental}}")
            out.append(symbols)
            out.append("")
            continue

        words = line.get("words", [])
        chords = sorted(line.get("chords", []), key=lambda c: c.get("startTimeMs", 0))
        pieces: List[str] = []
        for i, w in enumerate(words):
            for c in chords:
                if c.get("wordIndex") == i and c.get("anchor") == "before_word":
                    pieces.append(f"[{c['chordName']}]")
            text = w.get("text", "")
            placed_inside = [c for c in chords
                             if c.get("wordIndex") == i and c.get("anchor") == "on_word"]
            if not placed_inside:
                pieces.append(text)
            else:
                # Split the word so the bracket lands where the change does.
                cuts = sorted({
                    max(0, min(len(text), int(round(c.get("offsetRatio", 0.0) * len(text)))))
                    for c in placed_inside
                })
                by_cut: Dict[int, List[str]] = {}
                for c in placed_inside:
                    k = max(0, min(len(text),
                                   int(round(c.get("offsetRatio", 0.0) * len(text)))))
                    by_cut.setdefault(k, []).append(c["chordName"])
                buf, prev = [], 0
                for k in cuts:
                    buf.append(text[prev:k])
                    buf.append("".join(f"[{n}]" for n in by_cut[k]))
                    prev = k
                buf.append(text[prev:])
                pieces.append("".join(buf))
            pieces.append(" ")
        out.append("".join(pieces).rstrip())
        out.append("")

    return "\n".join(out).rstrip() + "\n"


# --- plain text --------------------------------------------------------------

def to_text(doc: ExportDocument) -> str:
    """Chord-above-lyric columns for LTR; inline brackets for RTL.

    The inline form is not a shortcut.  Column alignment genuinely cannot
    work for right-to-left text in a plain text file, because the file
    stores logical order and the viewer renders visual order, so the
    chord line and the lyric line are reordered differently.
    """
    out: List[str] = [doc.title, "=" * max(8, len(doc.title))]
    meta = []
    if doc.key_name:
        meta.append(f"Key: {doc.key_name}")
    if doc.tempo_bpm:
        meta.append(f"Tempo: {doc.tempo_bpm:.0f} BPM")
    if doc.transpose_semitones:
        meta.append(f"Transposed: {doc.transpose_semitones:+d}")
    if meta:
        out.append("  |  ".join(meta))
    out.append("")

    for line in doc.lines:
        chords = sorted(line.get("chords", []), key=lambda c: c.get("startTimeMs", 0))
        if line.get("isInstrumental"):
            out.append("   ".join(c["chordName"] for c in chords))
            out.append("")
            continue

        words = line.get("words", [])
        lyric = " ".join(w.get("text", "") for w in words)
        if not lyric.strip():
            continue

        if line_direction(lyric) == "rtl":
            pieces = []
            for i, w in enumerate(words):
                for c in chords:
                    if c.get("wordIndex") == i:
                        pieces.append(f"[{c['chordName']}]")
                pieces.append(w.get("text", ""))
            out.append(" ".join(pieces))
            out.append("")
            continue

        # LTR: build the chord line by column.
        positions: List[Tuple[int, str]] = []
        col = 0
        col_of_word: List[int] = []
        for w in words:
            col_of_word.append(col)
            col += len(w.get("text", "")) + 1

        for c in chords:
            wi = c.get("wordIndex")
            if wi is None or wi >= len(words):
                continue
            base = col_of_word[wi]
            inner = int(round(c.get("offsetRatio", 0.0) * len(words[wi].get("text", ""))))
            positions.append((base + inner, c["chordName"]))

        chord_line = ""
        for pos, name in sorted(positions):
            if len(chord_line) > pos:
                chord_line += " "
            else:
                chord_line += " " * (pos - len(chord_line))
            chord_line += name

        if chord_line.strip():
            out.append(chord_line)
        out.append(lyric)
        out.append("")

    return "\n".join(out).rstrip() + "\n"


# --- Markdown ---------------------------------------------------------------

def to_markdown(doc: ExportDocument) -> str:
    out = [f"# {doc.title}", ""]
    meta = []
    if doc.key_name:
        meta.append(f"**Key:** {doc.key_name}")
    if doc.tempo_bpm:
        meta.append(f"**Tempo:** {doc.tempo_bpm:.0f} BPM")
    if doc.transpose_semitones:
        meta.append(f"**Transposed:** {doc.transpose_semitones:+d} semitones")
    if meta:
        out += [" · ".join(meta), ""]

    out += ["```", to_text(doc).split("\n", 3)[-1].rstrip(), "```", ""]
    return "\n".join(out)


# --- JSON --------------------------------------------------------------------

def to_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


# --- PDF ----------------------------------------------------------------------

_HEBREW_CAPABLE_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
_HEBREW_CAPABLE_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def _register_fonts() -> Tuple[str, str]:
    """Register a Unicode font so Hebrew actually renders.

    reportlab's built-in fonts are Latin-1 only; Hebrew drawn with them
    comes out as empty boxes.  If no Unicode font can be found the
    exporter raises rather than producing a PDF full of blanks.
    """
    import os
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    regular = bold = None
    for path in _HEBREW_CAPABLE_FONTS:
        if os.path.isfile(path):
            pdfmetrics.registerFont(TTFont("CVSans", path))
            regular = "CVSans"
            break
    for path in _HEBREW_CAPABLE_BOLD:
        if os.path.isfile(path):
            pdfmetrics.registerFont(TTFont("CVSans-Bold", path))
            bold = "CVSans-Bold"
            break

    if regular is None:
        raise RuntimeError(
            "No Unicode TrueType font was found, so a PDF with Hebrew text "
            "cannot be produced. Install DejaVu Sans (fonts-dejavu-core) or "
            "point the exporter at another Unicode font."
        )
    return regular, bold or regular


def to_pdf(
    doc: ExportDocument,
    font_size: float = 11.0,
    chord_size: Optional[float] = None,
    page_size: str = "A4",
) -> bytes:
    """Render the chord sheet to PDF with correct bidirectional layout."""
    from reportlab.lib.pagesizes import A4, LETTER
    from reportlab.pdfgen import canvas as pdfcanvas
    from reportlab.pdfbase import pdfmetrics

    regular, bold = _register_fonts()
    chord_size = chord_size or max(8.0, font_size * 0.92)

    size = A4 if page_size.upper() == "A4" else LETTER
    page_w, page_h = size
    margin = 48.0
    usable_w = page_w - 2 * margin

    buf = io.BytesIO()
    c = pdfcanvas.Canvas(buf, pagesize=size)
    c.setTitle(doc.title)

    # Vertical budget for one block of (chord row + lyric row).
    #
    # Measured in points from the top of the block:
    #   chord baseline  = top - chord_size          (chord glyphs sit in that gap)
    #   lyric baseline  = chord baseline - gap_between
    #   next block top  = lyric baseline - descender - block_gap
    #
    # Deriving it this way rather than from a single "leading" number is
    # what keeps two consecutive lines from colliding: the earlier version
    # advanced by less than the block it had just drawn, so each lyric row
    # ran into the next line's chord row.
    gap_between = chord_size * 0.55 + font_size * 0.95
    descender = font_size * 0.26
    block_gap = font_size * 0.80
    block_height = chord_size + gap_between + descender + block_gap

    y = page_h - margin

    def new_page() -> float:
        c.showPage()
        return page_h - margin

    def need(height: float, cursor: float) -> float:
        if cursor - height < margin:
            return new_page()
        return cursor

    # --- header
    c.setFont(bold, font_size * 1.9)
    c.drawString(margin, y, _shape(doc.title))
    y -= font_size * 2.3

    meta = []
    if doc.key_name:
        meta.append(f"Key: {doc.key_name}")
    if doc.tempo_bpm:
        meta.append(f"Tempo: {doc.tempo_bpm:.0f} BPM")
    if doc.transpose_semitones:
        meta.append(f"Transposed {doc.transpose_semitones:+d} semitones")
    if meta:
        c.setFont(regular, font_size * 0.85)
        c.setFillGray(0.35)
        c.drawString(margin, y, "   |   ".join(meta))
        c.setFillGray(0.0)
        y -= font_size * 1.6

    c.setStrokeGray(0.8)
    c.line(margin, y, page_w - margin, y)
    y -= font_size * 1.2

    space_w = pdfmetrics.stringWidth(" ", regular, font_size)

    for line in doc.lines:
        chords = sorted(line.get("chords", []), key=lambda ch: ch.get("startTimeMs", 0))

        if line.get("isInstrumental"):
            inst_height = chord_size + descender + block_gap
            y = need(inst_height, y)
            c.setFont(bold, chord_size)
            c.setFillColorRGB(0.11, 0.42, 0.95)
            text = "   ".join(ch["chordName"] for ch in chords)
            c.drawString(margin, y - chord_size, text)
            c.setFillGray(0.0)
            y -= inst_height
            continue

        words = line.get("words", [])
        if not words:
            continue
        lyric_text = " ".join(w.get("text", "") for w in words)
        direction = line_direction(lyric_text)

        # Measure every word, then lay the line out.
        shaped = [_shape(w.get("text", "")) for w in words]
        widths = [pdfmetrics.stringWidth(s, regular, font_size) for s in shaped]

        total_w = sum(widths) + space_w * max(0, len(words) - 1)
        wrap = total_w > usable_w

        y = need(block_height, y)
        chord_y = y - chord_size
        lyric_y = chord_y - gap_between

        # x position of each word, as drawn.
        x_left: List[float] = []
        if direction == "rtl":
            cursor = page_w - margin
            for w_width in widths:
                cursor -= w_width
                x_left.append(cursor)
                cursor -= space_w
        else:
            cursor = margin
            for w_width in widths:
                x_left.append(cursor)
                cursor += w_width + space_w

        if wrap:
            # Shrink to fit rather than break the chord alignment across
            # two visual lines, which would need the chords re-anchored.
            scale = usable_w / total_w
            eff_size = max(6.5, font_size * scale)
            shaped = [_shape(w.get("text", "")) for w in words]
            widths = [pdfmetrics.stringWidth(s, regular, eff_size) for s in shaped]
            sp = pdfmetrics.stringWidth(" ", regular, eff_size)
            x_left = []
            if direction == "rtl":
                cursor = page_w - margin
                for w_width in widths:
                    cursor -= w_width
                    x_left.append(cursor)
                    cursor -= sp
            else:
                cursor = margin
                for w_width in widths:
                    x_left.append(cursor)
                    cursor += w_width + sp
        else:
            eff_size = font_size

        c.setFont(regular, eff_size)
        for s, x in zip(shaped, x_left):
            c.drawString(x, lyric_y, s)

        # Chord symbols, each above the point in the word where it starts.
        c.setFont(bold, chord_size)
        c.setFillColorRGB(0.11, 0.42, 0.95)
        for ch in chords:
            wi = ch.get("wordIndex")
            if wi is None or wi >= len(words):
                continue
            offset = float(ch.get("offsetRatio", 0.0))
            w_width = widths[wi]
            if ch.get("anchor") == "before_word":
                x = (x_left[wi] + w_width) if direction == "rtl" else x_left[wi]
            elif direction == "rtl":
                # Offset runs from the word's right edge leftwards.
                x = x_left[wi] + w_width * (1.0 - offset)
            else:
                x = x_left[wi] + w_width * offset
            name_w = pdfmetrics.stringWidth(ch["chordName"], bold, chord_size)
            x = max(margin, min(x, page_w - margin - name_w))
            c.drawString(x, chord_y, ch["chordName"])
        c.setFillGray(0.0)

        y = lyric_y - descender - block_gap

    if doc.generated_note:
        c.setFont(regular, 7.5)
        c.setFillGray(0.55)
        c.drawString(margin, margin * 0.55, doc.generated_note)

    c.save()
    return buf.getvalue()


# --- printable HTML ------------------------------------------------------------

def to_html(doc: ExportDocument) -> str:
    """A print-ready page. The browser handles bidi natively and correctly."""
    def esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;")
                 .replace(">", "&gt;").replace('"', "&quot;"))

    rows: List[str] = []
    for line in doc.lines:
        chords = sorted(line.get("chords", []), key=lambda c: c.get("startTimeMs", 0))
        if line.get("isInstrumental"):
            syms = "".join(f'<span class="ch">{esc(c["chordName"])}</span>' for c in chords)
            rows.append(f'<div class="line instrumental">{syms}</div>')
            continue

        words = line.get("words", [])
        if not words:
            continue
        lyric = " ".join(w.get("text", "") for w in words)
        direction = line_direction(lyric)

        by_word: Dict[int, List[Dict[str, Any]]] = {}
        for c in chords:
            by_word.setdefault(c.get("wordIndex", 0), []).append(c)

        cells: List[str] = []
        for i, w in enumerate(words):
            above = "".join(
                f'<span class="ch" style="--off:{c.get("offsetRatio", 0):.3f}">'
                f'{esc(c["chordName"])}</span>'
                for c in by_word.get(i, [])
            )
            cells.append(
                f'<span class="w"><span class="chords">{above}</span>'
                f'<span class="t">{esc(w.get("text", ""))}</span></span>'
            )
        rows.append(f'<div class="line" dir="{direction}">{"".join(cells)}</div>')

    meta = []
    if doc.key_name:
        meta.append(f"Key: {esc(doc.key_name)}")
    if doc.tempo_bpm:
        meta.append(f"Tempo: {doc.tempo_bpm:.0f} BPM")
    if doc.transpose_semitones:
        meta.append(f"Transposed {doc.transpose_semitones:+d}")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{esc(doc.title)}</title>
<style>
  :root {{ --ink:#111; --accent:#1b57d6; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,"Segoe UI",system-ui,"Noto Sans Hebrew",sans-serif;
         color:var(--ink); margin:0; padding:40px; line-height:1.2; }}
  h1 {{ font-size:28px; margin:0 0 4px; }}
  .meta {{ color:#666; font-size:13px; margin-bottom:22px;
           padding-bottom:14px; border-bottom:1px solid #ddd; }}
  .line {{ margin:0 0 20px; font-size:17px; }}
  .w {{ display:inline-block; position:relative; margin-inline-end:.42em;
        padding-top:1.5em; vertical-align:bottom; }}
  .chords {{ position:absolute; top:0; inset-inline-start:0; width:100%;
             height:1.4em; direction:ltr; }}
  .ch {{ position:absolute; inset-inline-start:calc(var(--off,0) * 100%);
         color:var(--accent); font-weight:700; font-size:14px;
         white-space:nowrap; font-family:ui-monospace,"SF Mono",Menlo,monospace; }}
  .instrumental .ch {{ position:static; margin-inline-end:1.1em; }}
  .instrumental {{ padding-top:0; }}
  @media print {{ body {{ padding:0; }} .line {{ break-inside:avoid; }} }}
</style></head>
<body>
<h1>{esc(doc.title)}</h1>
<div class="meta">{" &nbsp;|&nbsp; ".join(meta)}</div>
{chr(10).join(rows)}
</body></html>"""
