import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft, AlertTriangle, RefreshCw, Mic, Pencil, Eye } from "lucide-react";
import {
  api, ApiError, type Project, type SongView as SongViewData, type Settings,
} from "../lib/api";
import { useJobProgress } from "../hooks/useJobProgress";
import { usePlayer } from "../hooks/usePlayer";
import { AnalysisStages } from "../components/AnalysisStages";
import { SongView } from "../components/SongView";
import { Player } from "../components/Player";
import { TransposeBar } from "../components/TransposeBar";
import { ExportMenu } from "../components/ExportMenu";
import { SpectrumBars } from "../components/SpectrumBars";
import { useToast } from "../components/Toast";
import { formatTime } from "../lib/format";

export function Song({ settings }: { settings: Settings | null }) {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const toast = useToast();

  const [project, setProject] = useState<Project | null>(null);
  const [song, setSong] = useState<SongViewData | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [editMode, setEditMode] = useState(false);
  const [simplifiedChords, setSimplifiedChords] = useState(() => localStorage.getItem("chordvision.simplified") === "true");
  const [fontScale, setFontScale] = useState(() => { const saved = Number(localStorage.getItem("chordvision.sheetScale") ?? "1"); return Number.isFinite(saved) ? Math.min(1.6, Math.max(0.8, saved)) : 1; });
  useEffect(() => { localStorage.setItem("chordvision.simplified", String(simplifiedChords)); }, [simplifiedChords]);
  useEffect(() => { localStorage.setItem("chordvision.sheetScale", String(fontScale)); }, [fontScale]);

  const analysing =
    project?.status === "analyzing" || project?.status === "uploaded" || !project;
  const progress = useJobProgress(id, analysing);

  const player = usePlayer(project && project.status !== "failed" ? api.audioUrl(id) : null);

  const load = useCallback(async () => {
    try {
      const p = await api.getProject(id);
      setProject(p);
      if (p.status === "ready") setSong(await api.song(id));
    } catch (err) {
      setLoadError(err instanceof ApiError ? err.message : String(err));
    }
  }, [id]);

  useEffect(() => { void load(); }, [load]);

  // Reload once the worker reports it has finished.
  useEffect(() => {
    if (progress?.status === "succeeded" || progress?.status === "failed") {
      void load();
    }
  }, [progress?.status, load]);

  // Chord alternatives live on the project, placement lives on the song
  // view; the editor needs both, so they are merged here rather than in
  // the component.
  const lines = useMemo(() => {
    if (!song || !project) return [];
    const alts = new Map(project.chords.map((c) => [c.id, c.alternatives]));
    return song.lines.map((line) => ({
      ...line,
      chords: line.chords.map((c) => ({ ...c, alternatives: alts.get(c.chordId) ?? [] })),
    }));
  }, [song, project]);

  const refresh = useCallback(async () => {
    const [p, s] = await Promise.all([api.getProject(id), api.song(id)]);
    setProject(p);
    setSong(s);
  }, [id]);

  async function setTranspose(value: number) {
    if (!project) return;
    setProject({ ...project, transposeSemitones: value });
    try {
      await api.patchProject(id, { transposeSemitones: value });
      await refresh();
    } catch (err) {
      toast.show("error", "לא ניתן לשנות את הסולם",
        err instanceof ApiError ? err.message : String(err));
      await refresh();
    }
  }

  async function saveChord(chordId: string, name: string) {
    try {
      await api.patchChord(id, chordId, { chordName: name });
      await refresh();
      toast.show("ok", `האקורד שונה ל־${name}`);
    } catch (err) {
      toast.show("error", "לא ניתן לשמור את האקורד",
        err instanceof ApiError ? err.message : String(err));
    }
  }

  async function deleteChord(chordId: string) {
    try {
      await api.deleteChord(id, chordId);
      await refresh();
      toast.show("ok", "האקורד נמחק");
    } catch (err) {
      toast.show("error", "לא ניתן למחוק את האקורד",
        err instanceof ApiError ? err.message : String(err));
    }
  }

  async function addChordAt(ms: number) {
    try {
      await api.addChord(id, {
        chordName: "C",
        startTimeMs: Math.round(ms),
        endTimeMs: Math.round(ms) + 2000,
      });
      await refresh();
      toast.show("ok", "האקורד נוסף", "לחצו עליו כדי לבחור את הסימון הנכון.");
    } catch (err) {
      toast.show("error", "לא ניתן להוסיף אקורד",
        err instanceof ApiError ? err.message : String(err));
    }
  }

  async function editWord(word: { id: string }, text: string) {
    try {
      await api.patchWord(id, word.id, { text });
      await refresh();
    } catch (err) {
      toast.show("error", "לא ניתן לשמור את המילה",
        err instanceof ApiError ? err.message : String(err));
    }
  }

  async function reanalyze(withLyrics: boolean) {
    try {
      await api.reanalyze(id, withLyrics);
      setProject((p) => (p ? { ...p, status: "analyzing" } : p));
      toast.show("info", "הניתוח הופעל מחדש");
    } catch (err) {
      toast.show("error", "לא ניתן להפעיל מחדש את הניתוח",
        err instanceof ApiError ? err.message : String(err));
    }
  }

  if (loadError) {
    return (
      <div className="mx-auto max-w-2xl px-5 py-24 text-center">
        <AlertTriangle size={28} className="mx-auto text-danger mb-4" />
        <h1 className="text-xl font-bold text-ink">לא ניתן לפתוח את השיר</h1>
        <p className="text-muted mt-2">{loadError}</p>
        <Link to="/" className="btn-primary mt-6">חזרה להעלאת שיר</Link>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="mx-auto max-w-2xl px-5 py-24">
        <div className="h-7 w-52 rounded-lg bg-raised animate-pulse" />
        <div className="h-4 w-80 rounded-lg bg-raised/70 animate-pulse mt-4" />
        <div className="h-48 rounded-xl2 bg-raised/50 animate-pulse mt-8" />
      </div>
    );
  }

  // --- still analysing -----------------------------------------------
  if (project.status !== "ready") {
    const failed = project.status === "failed" || progress?.status === "failed";
    return (
      <div className="mx-auto max-w-2xl px-5 py-16">
        <Link to="/" className="btn-ghost -ms-3 mb-6">
          <ArrowLeft size={16} className="rtl:rotate-180" />
          חזרה
        </Link>

        <h1 className="text-2xl font-black text-ink">{project.title}</h1>
        <p className="text-muted mt-1">
          {formatTime(project.durationMs)} · {project.fileName}
        </p>

        {failed ? (
          <div className="panel p-6 mt-8 border-danger/40">
            <div className="flex items-start gap-3">
              <AlertTriangle size={20} className="text-danger mt-0.5 shrink-0" />
              <div>
                <h2 className="font-bold text-ink">הניתוח לא הסתיים</h2>
                <p className="text-sm text-muted mt-1.5 break-words">
                  {progress?.errorMessage ?? "תהליך הניתוח נעצר לפני שהפיק תוצאה."}
                </p>
                <button className="btn-outline mt-4" onClick={() => reanalyze(false)}>
                  <RefreshCw size={15} />
                  נסה שוב
                </button>
              </div>
            </div>
          </div>
        ) : (
          <motion.div
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
            className="panel p-6 mt-8"
          >
            <SpectrumBars active height={40} className="w-32 mb-6 opacity-70" />

            <div className="flex items-baseline gap-3 mb-5">
              <span className="text-3xl font-black text-ink tabular-nums">
                {Math.round((progress?.progress ?? 0) * 100)}%
              </span>
              <span className="text-sm text-muted">{progress?.message ?? "מתחיל לפעול"}</span>
            </div>

            <div className="h-1.5 rounded-full bg-base overflow-hidden mb-6">
              <motion.div
                className="h-full bg-glow rounded-full"
                animate={{ width: `${(progress?.progress ?? 0) * 100}%` }}
                transition={{ duration: 0.4, ease: "easeOut" }}
              />
            </div>

            <AnalysisStages stages={progress?.stages ?? []} progress={progress} />

            <p className="text-xs text-faint mt-6 leading-relaxed">
              The percentage is the worker's real position in the job, not a
              timer. You can leave this page; the analysis keeps running.
            </p>
          </motion.div>
        )}
      </div>
    );
  }

  // --- ready -----------------------------------------------------------
  const keyLabel = project.keyTonic ? `${project.keyTonic} ${project.keyMode === "major" ? "מז׳ור" : "מינור"}` : null;

  return (
    <div className="flex flex-col min-h-[calc(100vh-4rem)]">
      <div className="border-b border-line bg-base/85 backdrop-blur-xl sticky top-16 z-30">
        <div className="mx-auto max-w-5xl px-5 py-3 flex items-center gap-3 flex-wrap">
          <button onClick={() => navigate("/")} className="btn-ghost !px-2 -ms-2"
                  aria-label="Back">
            <ArrowLeft size={18} className="rtl:rotate-180" />
          </button>

          <div className="min-w-0 me-auto">
            <h1 className="font-bold text-ink truncate leading-tight">{project.title}</h1>
            <p className="text-xs text-muted">
              {keyLabel && (
                <span title={`רמת ודאות בזיהוי הסולם: ${Math.round((project.keyConfidence ?? 0) * 100)}%`}>
                  {keyLabel}
                </span>
              )}
              {project.tempoBpm && <>  ·  {Math.round(project.tempoBpm)} BPM</>}
              {project.tuningOffsetCents !== null &&
                Math.abs(project.tuningOffsetCents) > 12 && (
                  <>  ·  ההקלטה גבוהה/נמוכה ב־{" "}
                    {Math.abs(Math.round(project.tuningOffsetCents))} cents</>
                )}
            </p>
          </div>

          <TransposeBar
            semitones={project.transposeSemitones}
            originalKey={keyLabel}
            onChange={setTranspose}
          />

          <button
            className={editMode ? "btn-primary" : "btn-outline"}
            onClick={() => setEditMode((e) => !e)}
            title={editMode ? "סיום עריכה" : "עריכת אקורדים ומילים"}
          >
            {editMode ? <Eye size={16} /> : <Pencil size={16} />}
            <span className="hidden sm:inline">{editMode ? "סיום" : "עריכה"}</span>
          </button>

          <button className={simplifiedChords ? "btn-primary" : "btn-outline"} onClick={() => setSimplifiedChords((v) => !v)} title="הצגה פשוטה של האקורדים בלי לשנות את הנתונים המקוריים">{simplifiedChords ? "אקורדים מלאים" : "פישוט אקורדים"}</button>
          <div className="flex items-center gap-1 rounded-xl border border-line bg-base p-0.5" aria-label="גודל התצוגה"><span className="px-2 text-xs text-muted">גודל</span><button className="btn-ghost !px-2 !py-1" onClick={() => setFontScale((v) => Math.max(0.8, Math.round((v - 0.1) * 10) / 10))} disabled={fontScale <= 0.8} aria-label="הקטנת גודל">א−</button><span className="min-w-10 text-center text-xs tabular-nums">{Math.round(fontScale * 100)}%</span><button className="btn-ghost !px-2 !py-1" onClick={() => setFontScale((v) => Math.min(1.6, Math.round((v + 0.1) * 10) / 10))} disabled={fontScale >= 1.6} aria-label="הגדלת גודל">א+</button></div>
          <ExportMenu projectId={project.id} />
        </div>
      </div>

      <main className="flex-1 mx-auto max-w-3xl w-full px-5 pt-8">
        {!project.hasLyrics && (
          <div className="panel p-4 mb-8 flex items-start gap-3">
            <Mic size={17} className="text-muted mt-0.5 shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-sm text-ink font-medium">עדיין אין מילים</p>
              <p className="text-xs text-muted mt-1 leading-relaxed">
                The chords are here. To add the words, set up a transcription
                service in settings, then run the analysis again.
              </p>
            </div>
            {settings?.apiKeySet && settings.allowAudioUpload && (
              <button className="btn-outline !py-1.5 !px-3 text-xs shrink-0"
                      onClick={() => reanalyze(true)}>
                תמלול עכשיו
              </button>
            )}
          </div>
        )}

        {editMode && (
          <p className="text-xs text-muted mb-6">
            Click a chord to change it, double-click a word to retype it, and use
            the plus at the end of a line to add a chord.
          </p>
        )}

        <SongView
          lines={lines}
          positionMs={player.positionMs}
          notation={settings?.chordNotation ?? "sharps"}
          editable={editMode}
          onSeek={player.seek}
          onSaveChord={saveChord}
          onDeleteChord={deleteChord}
          onAddChordAt={addChordAt}
          onEditWord={editWord}
          simplified={simplifiedChords}
          fontScale={fontScale}
        />
      </main>

      <div className="sticky bottom-0 z-30">
        <Player player={player} peaks={project.waveformPeaks} beatsMs={project.beatTimesMs} />
      </div>
    </div>
  );
}
