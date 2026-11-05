export function formatTime(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) ms = 0;
  const total = Math.floor(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatDate(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const now = Date.now();
  const diff = now - d.getTime();
  if (diff < 60_000) return "הרגע";
  if (diff < 3_600_000) return `לפני ${Math.floor(diff / 60_000)} דקות`;
  if (diff < 86_400_000) return `לפני ${Math.floor(diff / 3_600_000)} שעות`;
  if (diff < 7 * 86_400_000) return `לפני ${Math.floor(diff / 86_400_000)} ימים`;
  return d.toLocaleDateString("he-IL", { day: "numeric", month: "short", year: "numeric" });
}

const RTL = /[֐-׿؀-ۿ܀-ݏיִ-﷿ﹰ-﻿]/;

/** Decide a line's writing direction from the text itself.
 *  The app has to render Hebrew and English side by side in one song,
 *  so direction is a per-line property, never a global setting. */
export function textDirection(text: string): "rtl" | "ltr" {
  let rtl = 0, ltr = 0;
  for (const ch of text) {
    if (RTL.test(ch)) rtl++;
    else if (/[A-Za-zÀ-ɏ]/.test(ch)) ltr++;
  }
  return rtl > ltr ? "rtl" : "ltr";
}

/** How much to trust a confidence figure, in words a musician can act on. */
export function confidenceBand(c: number): { label: string; tone: "high" | "medium" | "low" } {
  if (c >= 0.7) return { label: "זיהוי בטוח", tone: "high" };
  if (c >= 0.4) return { label: "זיהוי סביר", tone: "medium" };
  return { label: "זיהוי לא ודאי", tone: "low" };
}

export const TIMING_SOURCE_LABEL: Record<string, string> = {
  provider_word: "תזמון שנקבע על ידי שירות התמלול",
  forced_alignment: "תזמון שהותאם לשמע",
  estimated: "תזמון משוער לפי חלוקת המשפט",
  user: "תזמון שהוגדר ידנית",
};

export const PITCH_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
export const PITCH_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];

export const CHORD_QUALITIES = [
  { id: "", label: "major" },
  { id: "m", label: "minor" },
  { id: "7", label: "dominant 7" },
  { id: "maj7", label: "major 7" },
  { id: "m7", label: "minor 7" },
  { id: "sus2", label: "sus2" },
  { id: "sus4", label: "sus4" },
  { id: "6", label: "6" },
  { id: "m6", label: "minor 6" },
  { id: "dim", label: "diminished" },
  { id: "aug", label: "augmented" },
  { id: "m7b5", label: "half diminished" },
  { id: "dim7", label: "diminished 7" },
  { id: "add9", label: "add9" },
  { id: "9", label: "9" },
  { id: "maj9", label: "major 9" },
  { id: "m9", label: "minor 9" },
  { id: "11", label: "11" },
  { id: "13", label: "13" },
];

/** Split a chord symbol into its parts, for the editor's pickers. */
export function splitChord(symbol: string): { root: string; quality: string; bass: string | null } {
  const m = /^([A-G][#b]?)(.*?)(?:\/([A-G][#b]?))?$/.exec(symbol.trim());
  if (!m) return { root: "C", quality: "", bass: null };
  return { root: m[1], quality: m[2] ?? "", bass: m[3] ?? null };
}

export function joinChord(root: string, quality: string, bass: string | null): string {
  return root + quality + (bass && bass !== root ? `/${bass}` : "");
}

/** Simplifies notation for display only; source chords remain unchanged. */
export function simplifyChordSymbol(symbol: string): string {
  const { root, quality, bass } = splitChord(symbol);
  const q = quality.toLowerCase();
  let simple = "";
  if (q.includes("dim") || q.includes("m7b5") || q.includes("ø")) simple = "dim";
  else if (q.includes("aug") || q.startsWith("+")) simple = "aug";
  else if (q.startsWith("sus")) simple = "sus";
  else if (q.startsWith("m") && !q.startsWith("maj")) simple = "m";
  return root + simple + (bass && bass !== root ? `/${bass}` : "");
}
