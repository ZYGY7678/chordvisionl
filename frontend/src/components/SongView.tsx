import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Plus, Music2 } from "lucide-react";
import type { SongLine, PlacedChord, Word, ChordAlternative } from "../lib/api";
import { textDirection, TIMING_SOURCE_LABEL, simplifyChordSymbol } from "../lib/format";
import { ChordPopover } from "./ChordPopover";

/**
 * Chords above lyrics, following the audio.
 *
 * The placement comes from the server, which anchored each chord to a
 * word and gave an offset for where inside that word the change happens.
 * Here that offset becomes a percentage across the word's own box, so
 * the symbol sits over the syllable rather than over the word's start.
 *
 * Direction is decided per line from its own text, so a song with a
 * Hebrew verse and an English chorus lays both out correctly. The chord
 * row inside each word is always left-to-right, because chord symbols
 * are Latin whatever the lyric is doing - without that, "G/B" renders as
 * "B/G" on a Hebrew line.
 */
export function SongView({
  lines,
  positionMs,
  notation,
  editable,
  onSeek,
  onSaveChord,
  onDeleteChord,
  onAddChordAt,
  onEditWord,
  simplified,
  fontScale,
}: {
  lines: SongLine[];
  positionMs: number;
  notation: "sharps" | "flats";
  simplified: boolean;
  fontScale: number;
  editable: boolean;
  onSeek: (ms: number) => void;
  onSaveChord: (chordId: string, name: string) => void;
  onDeleteChord: (chordId: string) => void;
  onAddChordAt: (ms: number) => void;
  onEditWord: (word: Word, text: string) => void;
}) {
  const [openChord, setOpenChord] = useState<string | null>(null);
  const [editingWord, setEditingWord] = useState<string | null>(null);
  const activeRef = useRef<HTMLDivElement | null>(null);
  const [follow, setFollow] = useState(true);

  const activeLineId = useMemo(() => {
    let found: number | null = null;
    for (const line of lines) {
      if (positionMs >= line.startTimeMs && positionMs < line.endTimeMs) return line.lineId;
      if (positionMs >= line.startTimeMs) found = line.lineId;
    }
    return found;
  }, [lines, positionMs]);

  // Dimming the inactive lines is a cue about *where the music is*, so it
  // only makes sense once the music is somewhere. Before playback starts
  // nothing is active, and dimming every line then just makes the whole
  // sheet look disabled.
  const following = activeLineId !== null;

  // Auto-scroll, but stop the moment the user scrolls themselves: a page
  // that keeps yanking itself back while someone is reading ahead is
  // worse than one that does not follow at all.
  useEffect(() => {
    if (!follow || !activeRef.current) return;
    activeRef.current.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [activeLineId, follow]);

  useEffect(() => {
    let timer: number | null = null;
    const onWheel = () => {
      setFollow(false);
      if (timer) window.clearTimeout(timer);
      timer = window.setTimeout(() => setFollow(true), 6000);
    };
    window.addEventListener("wheel", onWheel, { passive: true });
    window.addEventListener("touchmove", onWheel, { passive: true });
    return () => {
      window.removeEventListener("wheel", onWheel);
      window.removeEventListener("touchmove", onWheel);
      if (timer) window.clearTimeout(timer);
    };
  }, []);

  if (lines.length === 0) {
    return (
      <div className="py-20 text-center">
        <Music2 size={28} className="mx-auto text-faint mb-3" />
        <p className="text-muted">לא זוהו אקורדים בהקלטה הזאת.</p>
      </div>
    );
  }

  return (
    <div className="space-y-7 pb-10">
      {lines.map((line) => {
        const isActive = line.lineId === activeLineId;
        const lyric = line.words.map((w) => w.text).join(" ");
        const dir = line.isInstrumental ? "ltr" : textDirection(lyric);

        return (
          <div
            key={line.lineId}
            ref={isActive ? activeRef : undefined}
            className={`transition-opacity duration-500 ${
              !following || isActive ? "opacity-100" : "opacity-50"
            }`}
          >
            {line.isInstrumental ? (
              <div className="flex flex-wrap items-center gap-x-5 gap-y-2" dir="ltr">
                <span className="text-xs text-faint shrink-0">קטע אינסטרומנטלי</span>
                {line.chords.map((c) => (
                  <ChordChip
                    key={c.chordId}
                    chord={c}
                    active={positionMs >= c.startTimeMs && positionMs < c.endTimeMs}
                    notation={notation}
                    simplified={simplified}
                    fontScale={fontScale}
                    editable={editable}
                    open={openChord === c.chordId}
                    onOpen={() => setOpenChord(openChord === c.chordId ? null : c.chordId)}
                    onClose={() => setOpenChord(null)}
                    onSeek={onSeek}
                    onSave={onSaveChord}
                    onDelete={onDeleteChord}
                    inline
                  />
                ))}
              </div>
            ) : (
              <div dir={dir} className="flex flex-wrap items-end gap-x-1.5 gap-y-5">
                {line.words.map((word, i) => {
                  const chords = line.chords.filter((c) => c.wordIndex === i);
                  const wordActive =
                    positionMs >= word.startTimeMs && positionMs < word.endTimeMs;
                  return (
                    <span key={word.id} className="relative inline-block pt-9">
                      <span className="absolute top-0 inset-x-0 h-8" dir="ltr">
                        {chords.map((c) => (
                          <ChordChip
                            key={c.chordId}
                            chord={c}
                            active={positionMs >= c.startTimeMs && positionMs < c.endTimeMs}
                            notation={notation}
                            simplified={simplified}
                            fontScale={fontScale}
                            editable={editable}
                            open={openChord === c.chordId}
                            onOpen={() => setOpenChord(openChord === c.chordId ? null : c.chordId)}
                            onClose={() => setOpenChord(null)}
                            onSeek={onSeek}
                            onSave={onSaveChord}
                            onDelete={onDeleteChord}
                            positioned={dir}
                          />
                        ))}
                      </span>

                      {editable && editingWord === word.id ? (
                        <input
                          autoFocus
                          defaultValue={word.text}
                          dir={dir}
                          className="field !py-0.5 !px-2 !text-xl w-[9ch]"
                          onBlur={(e) => {
                            if (e.target.value !== word.text) onEditWord(word, e.target.value);
                            setEditingWord(null);
                          }}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") (e.target as HTMLInputElement).blur();
                            if (e.key === "Escape") setEditingWord(null);
                          }}
                        />
                      ) : (
                        <button
                          onClick={() => onSeek(word.startTimeMs)}
                          onDoubleClick={() => editable && setEditingWord(word.id)}
                          title={`${TIMING_SOURCE_LABEL[word.timingSource] ?? ""}${
                            editable ? " · double-click to edit" : ""
                          }`}
                          className={`text-xl leading-snug rounded px-0.5 transition-colors
                                      ${wordActive
                                        ? "bg-glow/20 text-ink"
                                        : "text-ink/90 hover:bg-raised"}
                                      ${word.timingSource === "estimated"
                                        ? "decoration-dotted underline decoration-faint/60 underline-offset-4"
                                        : ""}`}
                        >
                          {word.text}
                        </button>
                      )}
                    </span>
                  );
                })}

                {editable && (
                  <button
                    onClick={() => onAddChordAt(line.startTimeMs)}
                    className="self-end mb-1 p-1 rounded-lg text-faint hover:text-glow hover:bg-raised transition"
                    title="הוספת אקורד לשורה"
                    aria-label="הוספת אקורד לשורה"
                  >
                    <Plus size={15} />
                  </button>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function ChordChip({
  chord, active, notation, simplified, fontScale, editable, open, inline, positioned,
  onOpen, onClose, onSeek, onSave, onDelete,
}: {
  chord: PlacedChord;
  active: boolean;
  notation: "sharps" | "flats";
  editable: boolean;
  open: boolean;
  inline?: boolean;
  positioned?: "rtl" | "ltr";
  onOpen: () => void;
  onClose: () => void;
  onSeek: (ms: number) => void;
  onSave: (id: string, name: string) => void;
  onDelete: (id: string) => void;
}) {
  // offsetRatio runs from the start of the word in reading order, so on a
  // right-to-left line it is measured from the word's right edge.
  const style: React.CSSProperties = inline
    ? {}
    : positioned === "rtl"
    ? { right: `${chord.offsetRatio * 100}%` }
    : { left: `${chord.offsetRatio * 100}%` };

  const unsure = chord.confidence < 0.4;
  const displayName = simplified ? simplifyChordSymbol(chord.chordName) : chord.chordName;

  return (
    <span className={inline ? "relative inline-block" : "absolute"} style={{ ...style, fontSize: `${fontScale}em` }}>
      <button
        onClick={(e) => {
          e.stopPropagation();
          if (editable) onOpen();
          else onSeek(chord.startTimeMs);
        }}
        onDoubleClick={() => onSeek(chord.startTimeMs)}
        title={
          `${chord.chordName} — ${Math.round(chord.confidence * 100)}% confident` +
          (chord.isEdited ? " (edited by you)" : "") +
          (editable ? "\nClick to edit, double-click to jump here" : "")
        }
        className={`chord-type whitespace-nowrap rounded px-1 transition-all duration-150
                    ${active ? "text-[1.08em] drop-shadow-[0_0_10px_rgb(var(--glow)/0.55)]" : ""}
                    ${unsure ? "opacity-70" : ""}
                    ${editable ? "hover:bg-glow/15" : ""}`}
      >
        {displayName}
        {unsure && <span className="text-faint align-super text-[0.6em] ms-0.5">?</span>}
      </button>

      {active && !inline && (
        <motion.span
          layoutId="chord-underline"
          className="absolute -bottom-0.5 inset-x-1 h-[2px] rounded-full bg-glow"
          transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
        />
      )}

      {open && editable && (
        <ChordPopover
          chordName={chord.chordName}
          alternatives={(chord as unknown as { alternatives?: ChordAlternative[] }).alternatives ?? []}
          confidence={chord.confidence}
          notation={notation}
          onSave={(name) => { onSave(chord.chordId, name); onClose(); }}
          onDelete={() => { onDelete(chord.chordId); onClose(); }}
          onClose={onClose}
        />
      )}
    </span>
  );
}
