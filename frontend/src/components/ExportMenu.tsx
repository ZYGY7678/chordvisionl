import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Download, Printer, FileText, FileJson, FileCode, FileType2 } from "lucide-react";
import { api } from "../lib/api";

const FORMATS = [
  { id: "pdf", label: "PDF", hint: "Chords over lyrics, ready to print", icon: FileType2 },
  { id: "chordpro", label: "ChordPro", hint: "Opens in any chord app", icon: FileCode },
  { id: "txt", label: "Plain text", hint: "Monospaced chord sheet", icon: FileText },
  { id: "md", label: "Markdown", hint: "For notes and wikis", icon: FileText },
  { id: "json", label: "JSON", hint: "Every timestamp and confidence", icon: FileJson },
];

export function ExportMenu({ projectId }: { projectId: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, []);

  function download(fmt: string) {
    const a = document.createElement("a");
    a.href = api.exportUrl(projectId, fmt);
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setOpen(false);
  }

  function print() {
    // The printable page is server-rendered HTML, so the browser does the
    // bidirectional layout itself - which it does better than any PDF
    // library, for a song that mixes Hebrew and English.
    const w = window.open(api.exportUrl(projectId, "html"), "_blank", "noopener");
    if (w) w.addEventListener("load", () => w.print());
    setOpen(false);
  }

  return (
    <div className="relative" ref={ref}>
      <button className="btn-outline" onClick={() => setOpen((o) => !o)}>
        <Download size={16} />
        <span className="hidden sm:inline">Export</span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -6, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.98 }}
            transition={{ duration: 0.15, ease: [0.22, 1, 0.36, 1] }}
            className="absolute end-0 top-full mt-2 w-72 panel shadow-deep p-1.5 z-50"
          >
            {FORMATS.map((f) => (
              <button
                key={f.id}
                onClick={() => download(f.id)}
                className="w-full flex items-start gap-3 px-3 py-2.5 rounded-xl hover:bg-raised transition-colors text-start"
              >
                <f.icon size={16} className="text-muted mt-0.5 shrink-0" />
                <span className="min-w-0">
                  <span className="block text-sm text-ink font-medium">{f.label}</span>
                  <span className="block text-xs text-muted">{f.hint}</span>
                </span>
              </button>
            ))}
            <div className="h-px bg-line my-1.5 mx-3" />
            <button
              onClick={print}
              className="w-full flex items-start gap-3 px-3 py-2.5 rounded-xl hover:bg-raised transition-colors text-start"
            >
              <Printer size={16} className="text-muted mt-0.5 shrink-0" />
              <span>
                <span className="block text-sm text-ink font-medium">Print</span>
                <span className="block text-xs text-muted">Opens a print-ready page</span>
              </span>
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
