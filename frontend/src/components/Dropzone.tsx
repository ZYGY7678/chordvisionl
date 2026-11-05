import { useCallback, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Music4, X, Upload } from "lucide-react";
import { SpectrumBars } from "./SpectrumBars";
import { formatBytes, formatTime } from "../lib/format";

export interface PickedFile {
  file: File;
  durationMs: number | null;
}

export function Dropzone({
  formats,
  maxBytes,
  picked,
  onPick,
  onClear,
  disabled,
}: {
  formats: string[];
  maxBytes: number;
  picked: PickedFile | null;
  onPick: (p: PickedFile) => void;
  onClear: () => void;
  disabled?: boolean;
}) {
  const [over, setOver] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept = useCallback(
    async (file: File) => {
      setProblem(null);
      const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
      if (formats.length && !formats.includes(ext)) {
        setProblem(`${file.name} isn't a supported format. Use ${formats.join(", ")}.`);
        return;
      }
      if (file.size > maxBytes) {
        setProblem(
          `${file.name} is ${formatBytes(file.size)}. The limit is ${formatBytes(maxBytes)}.`,
        );
        return;
      }

      // Read the duration locally so the card can show it before upload.
      // The server checks the real duration again; this is for the user.
      const durationMs = await new Promise<number | null>((resolve) => {
        const url = URL.createObjectURL(file);
        const audio = new Audio();
        const done = (v: number | null) => {
          URL.revokeObjectURL(url);
          resolve(v);
        };
        audio.onloadedmetadata = () =>
          done(Number.isFinite(audio.duration) ? audio.duration * 1000 : null);
        audio.onerror = () => done(null);
        audio.src = url;
        window.setTimeout(() => done(null), 4000);
      });

      onPick({ file, durationMs });
    },
    [formats, maxBytes, onPick],
  );

  if (picked) {
    return (
      <motion.div
        layout
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
        className="panel p-5 flex items-center gap-4"
      >
        <div className="size-12 rounded-xl bg-glow/15 grid place-items-center shrink-0">
          <Music4 size={22} className="text-glow" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="font-medium text-ink truncate">{picked.file.name}</p>
          <p className="text-sm text-muted mt-0.5">
            {formatBytes(picked.file.size)}
            {picked.durationMs ? ` · ${formatTime(picked.durationMs)}` : ""}
          </p>
        </div>
        <button className="btn-ghost !px-2.5" onClick={onClear} aria-label="Remove file">
          <X size={18} />
        </button>
      </motion.div>
    );
  }

  return (
    <div>
      <motion.div
        animate={{ scale: over ? 1.01 : 1 }}
        transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          if (disabled) return;
          const f = e.dataTransfer.files?.[0];
          if (f) void accept(f);
        }}
        onClick={() => !disabled && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        className={`chroma-grid relative rounded-xl3 border-2 border-dashed p-10 sm:p-14
                    text-center cursor-pointer transition-colors duration-200
                    ${over ? "border-glow bg-glow/5" : "border-line hover:border-faint bg-surface/40"}
                    ${disabled ? "opacity-50 pointer-events-none" : ""}`}
      >
        <SpectrumBars active={over} height={44} className="w-40 mx-auto mb-7 opacity-80" />
        <p className="text-xl font-bold text-ink">
          {over ? "Let go to add it" : "Drop a song here"}
        </p>
        <p className="text-muted mt-2">
          or <span className="text-glow font-medium">choose a file</span> from your device
        </p>
        <p className="text-xs text-faint mt-6 font-medium">
          {formats.join("  ")}
        </p>
        <input
          ref={inputRef}
          type="file"
          className="sr-only"
          accept="audio/*"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void accept(f);
            e.target.value = "";
          }}
        />
      </motion.div>

      <AnimatePresence>
        {problem && (
          <motion.p
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="text-sm text-danger mt-3"
          >
            {problem}
          </motion.p>
        )}
      </AnimatePresence>
    </div>
  );
}
