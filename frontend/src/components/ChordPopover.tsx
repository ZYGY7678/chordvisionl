import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Trash2, Check } from "lucide-react";
import { PITCH_SHARP, PITCH_FLAT, CHORD_QUALITIES, splitChord, joinChord } from "../lib/format";
import type { ChordAlternative } from "../lib/api";

/**
 * The chord editor: a root, a quality, an optional slash bass, and the
 * alternatives the engine actually considered.
 *
 * Showing the alternatives is the point. When the engine reports C with
 * 48% and Cmaj7 with 38%, the player can hear which one it is in a second
 * and the interface should make that a single click rather than a hunt
 * through a dropdown.
 */
export function ChordPopover({
  chordName,
  alternatives,
  confidence,
  notation,
  onSave,
  onDelete,
  onClose,
}: {
  chordName: string;
  alternatives: ChordAlternative[];
  confidence: number;
  notation: "sharps" | "flats";
  onSave: (name: string) => void;
  onDelete: () => void;
  onClose: () => void;
}) {
  const initial = splitChord(chordName);
  const [root, setRoot] = useState(initial.root);
  const [quality, setQuality] = useState(initial.quality);
  const [bass, setBass] = useState(initial.bass);
  const ref = useRef<HTMLDivElement>(null);

  const notes = notation === "flats" ? PITCH_FLAT : PITCH_SHARP;
  const preview = joinChord(root, quality, bass);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) onSave(preview);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [onClose, onSave, preview]);

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: -6, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.16, ease: [0.22, 1, 0.36, 1] }}
      className="absolute z-50 top-full mt-2 start-0 w-[21rem] panel shadow-deep p-4"
      dir="ltr"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="flex items-baseline gap-2 mb-3">
        <span className="chord-type text-2xl">{preview}</span>
        <span className="text-xs text-faint ms-auto">
          רמת ודאות הזיהוי {Math.round(confidence * 100)}%
        </span>
      </div>

      <div className="grid grid-cols-6 gap-1 mb-3">
        {notes.map((n) => (
          <button
            key={n}
            onClick={() => setRoot(n)}
            className={`py-1.5 rounded-lg text-sm font-chord transition-colors
                        ${root === n ? "bg-glow text-[rgb(var(--base))] font-bold"
                                     : "bg-base text-muted hover:text-ink hover:bg-raised"}`}
          >
            {n}
          </button>
        ))}
      </div>

      <div className="flex flex-wrap gap-1 mb-3 max-h-28 overflow-y-auto">
        {CHORD_QUALITIES.map((q) => (
          <button
            key={q.id}
            onClick={() => setQuality(q.id)}
            title={q.label}
            className={`px-2 py-1 rounded-lg text-xs transition-colors
                        ${quality === q.id ? "bg-raised text-ink font-bold border border-glow/50"
                                           : "bg-base text-muted hover:text-ink border border-transparent"}`}
          >
            {q.id === "" ? "maj" : q.id}
          </button>
        ))}
      </div>

      <label className="flex items-center gap-2 mb-3">
        <span className="text-xs text-muted w-20">צליל בס</span>
        <select
          className="field !py-1.5 !text-sm flex-1"
          value={bass ?? ""}
          onChange={(e) => setBass(e.target.value || null)}
        >
          <option value="">כמו צליל הבסיס</option>
          {notes.map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
      </label>

      {alternatives.length > 0 && (
        <div className="mb-3 pt-3 border-t border-line">
          <p className="text-xs text-muted mb-2">אפשרויות נוספות שהמנוע שקל</p>
          <div className="flex flex-wrap gap-1.5">
            {alternatives.map((a) => (
              <button
                key={a.chordName}
                onClick={() => {
                  const s = splitChord(a.chordName);
                  setRoot(s.root); setQuality(s.quality); setBass(s.bass);
                }}
                className="chip hover:!text-ink hover:!border-faint"
              >
                <span className="font-chord">{a.chordName}</span>
                <span className="text-faint tabular-nums">{Math.round(a.confidence * 100)}%</span>
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="flex gap-2">
        <button className="btn-primary flex-1" onClick={() => onSave(preview)}>
          <Check size={15} />
          שמירת האקורד
        </button>
        <button className="btn-danger !px-3" onClick={onDelete} aria-label="מחיקת האקורד">
          <Trash2 size={15} />
        </button>
      </div>
    </motion.div>
  );
}
