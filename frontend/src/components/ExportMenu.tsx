import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Download, Printer, FileText, FileJson, FileCode, FileType2, Loader2 } from "lucide-react";
import { api, ApiError } from "../lib/api";
import { useToast } from "./Toast";
const FORMATS = [
  { id: "pdf", label: "PDF להדפסה", hint: "אקורדים מעל המילים, מוכן להדפסה", icon: FileType2 },
  { id: "chordpro", label: "ChordPro", hint: "פתיחה באפליקציות אקורדים", icon: FileCode },
  { id: "txt", label: "טקסט פשוט", hint: "דף אקורדים טקסטואלי", icon: FileText },
  { id: "md", label: "Markdown", hint: "לפתקים ולוויקי", icon: FileText },
  { id: "json", label: "JSON", hint: "כל הזמנים ורמות הוודאות", icon: FileJson },
];
function failureMessage(body: string, status: number) {
  try { const parsed = JSON.parse(body); if (typeof parsed?.detail === "string") return parsed.detail; } catch { /* fallback */ }
  return status === 401 || status === 403 ? "ההתחברות פגה. רעננו את העמוד ונסו שוב." : `השרת לא הצליח להכין את הקובץ (שגיאה ${status}).`;
}
function filenameFromHeader(header: string | null, fallback: string) {
  const encoded = header?.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
  if (encoded) { try { return decodeURIComponent(encoded.replace(/\"/g, "")); } catch { /* fallback */ } }
  return header?.match(/filename="?([^";]+)"?/i)?.[1] || fallback;
}
export function ExportMenu({ projectId }: { projectId: string }) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [pdfFontSize, setPdfFontSize] = useState(11);
  const ref = useRef<HTMLDivElement>(null);
  const toast = useToast();
  useEffect(() => {
    const onDown = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, []);
  async function download(fmt: string) {
    setBusy(true); setOpen(false);
    try {
      const response = await fetch(api.exportUrl(projectId, fmt, fmt === "pdf" ? pdfFontSize : undefined), { credentials: "include" });
      if (!response.ok) throw new ApiError(failureMessage(await response.text(), response.status), response.status);
      const blob = await response.blob();
      if (!blob.size) throw new Error("הקובץ שהתקבל ריק.");
      const fallback = `ChordVision-${projectId}.${fmt === "chordpro" ? "pro" : fmt}`;
      const filename = filenameFromHeader(response.headers.get("Content-Disposition"), fallback);
      const objectUrl = URL.createObjectURL(blob);
      const link = document.createElement("a"); link.href = objectUrl; link.download = filename;
      document.body.appendChild(link); link.click(); link.remove();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000);
      toast.show("ok", "הקובץ מוכן", `ההורדה של ${filename} התחילה.`);
    } catch (error) {
      toast.show("error", "הורדת המסמך נכשלה", error instanceof Error ? error.message : String(error));
    } finally { setBusy(false); }
  }
  async function print() {
    const popup = window.open("", "_blank"); setOpen(false);
    if (!popup) { toast.show("error", "לא ניתן לפתוח חלון הדפסה", "יש לאפשר חלונות קופצים עבור האתר."); return; }
    setBusy(true);
    try {
      const response = await fetch(api.exportUrl(projectId, "html"), { credentials: "include" });
      if (!response.ok) throw new ApiError(failureMessage(await response.text(), response.status), response.status);
      popup.document.open(); popup.document.write(await response.text()); popup.document.close(); popup.focus();
      window.setTimeout(() => popup.print(), 500);
    } catch (error) {
      popup.close(); toast.show("error", "ההדפסה נכשלה", error instanceof Error ? error.message : String(error));
    } finally { setBusy(false); }
  }
  return (
    <div className="relative" ref={ref}>
      <button className="btn-outline" onClick={() => setOpen((o) => !o)} disabled={busy}>
        {busy ? <Loader2 size={16} className="animate-spin" /> : <Download size={16} />}
        <span className="hidden sm:inline">{busy ? "מכין קובץ…" : "הורדה"}</span>
      </button>
      <AnimatePresence>
        {open && <motion.div initial={{ opacity: 0, y: -6, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -4, scale: 0.98 }} transition={{ duration: 0.15, ease: [0.22, 1, 0.36, 1] }} className="absolute end-0 top-full mt-2 w-80 max-w-[calc(100vw-2rem)] panel shadow-deep p-1.5 z-50">
          {FORMATS.map((format) => <button key={format.id} onClick={() => void download(format.id)} className="w-full flex items-start gap-3 px-3 py-2.5 rounded-xl hover:bg-raised transition-colors text-start">
            <format.icon size={16} className="text-muted mt-0.5 shrink-0" /><span className="min-w-0"><span className="block text-sm text-ink font-medium">{format.label}</span><span className="block text-xs text-muted">{format.hint}</span></span>
          </button>)}
          <div className="border-t border-line my-1.5 px-3 pt-2"><label className="flex items-center justify-between gap-3 text-xs text-muted">גודל גופן ב־PDF <span className="font-bold text-ink tabular-nums">{pdfFontSize}</span></label>
            <input className="w-full mt-2 accent-[rgb(var(--glow))]" type="range" min={8} max={18} step={1} value={pdfFontSize} onChange={(event) => setPdfFontSize(Number(event.target.value))} aria-label="גודל גופן ב־PDF" /></div>
          <button onClick={() => void print()} className="w-full flex items-start gap-3 px-3 py-2.5 rounded-xl hover:bg-raised transition-colors text-start"><Printer size={16} className="text-muted mt-0.5 shrink-0" /><span><span className="block text-sm text-ink font-medium">הדפסה</span><span className="block text-xs text-muted">פתיחת דף מוכן להדפסה</span></span></button>
        </motion.div>}
      </AnimatePresence>
    </div>
  );
}
