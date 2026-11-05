import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight, Loader2, Mic } from "lucide-react";
import { Dropzone, type PickedFile } from "../components/Dropzone";
import { ProjectCard } from "../components/ProjectCard";
import { api, ApiError, type Health, type ProjectSummary, type Settings } from "../lib/api";
import { useToast } from "../components/Toast";

export function Home({
  health,
  settings,
  onOpenSettings,
}: {
  health: Health | null;
  settings: Settings | null;
  onOpenSettings: () => void;
}) {
  const [picked, setPicked] = useState<PickedFile | null>(null);
  const [title, setTitle] = useState("");
  const [transcribe, setTranscribe] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadFraction, setUploadFraction] = useState(0);
  const [recent, setRecent] = useState<ProjectSummary[]>([]);
  const navigate = useNavigate();
  const toast = useToast();

  useEffect(() => {
    api.listProjects().then((p) => setRecent(p.slice(0, 6))).catch(() => {});
  }, []);

  const canTranscribe = Boolean(settings?.apiKeySet && settings?.allowAudioUpload);

  async function start() {
    if (!picked) return;
    setUploading(true);
    setUploadFraction(0);
    try {
      const res = await api.upload(
        picked.file,
        title.trim() || picked.file.name.replace(/\.[^.]+$/, ""),
        transcribe,
        setUploadFraction,
      );
      navigate(`/song/${res.project.id}`);
    } catch (err) {
      const message = err instanceof ApiError ? err.message : String(err);
      toast.show("error", "Upload failed", message);
      setUploading(false);
    }
  }

  async function remove(id: string) {
    try {
      await api.deleteProject(id);
      setRecent((r) => r.filter((p) => p.id !== id));
      toast.show("ok", "Project deleted");
    } catch (err) {
      toast.show("error", "Could not delete the project",
        err instanceof ApiError ? err.message : String(err));
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-5 pb-24">
      {!health?.persistence?.ready && (
        <div className="mt-8 panel p-4 text-sm text-muted">
          <strong className="text-ink">הזיכרון הקבוע עדיין לא הושלם.</strong>
          <p className="mt-1">כדי שהשירים והמפתח של Gemini יישמרו גם אחרי פריסה מחדש, יש להשלים ב־Render את כתובת מסד הנתונים ואת מפתח האחסון הסודי של Supabase. עד אז האחסון עדיין זמני.</p>
        </div>
      )}
      <section className="pt-16 sm:pt-24 pb-10">
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        >
          <h1 className="text-[clamp(2rem,4.6vw,3.1rem)] font-black leading-[1.08] tracking-tight text-ink max-w-[19ch] text-balance">
            שומעים את האקורדים, קוראים את המילים — הכול מסונכרן להקלטה.
          </h1>
          <p className="mt-5 text-lg text-muted max-w-[58ch] leading-relaxed">
            מעלים שיר ו־ChordVision מזהה את ההרמוניה מתוך השמע, מציב את האקורדים מעל המילים ומסנכרן כל שינוי למקום שבו הוא מתרחש.
          </p>
        </motion.div>
      </section>

      <motion.section
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.08, ease: [0.22, 1, 0.36, 1] }}
      >
        <Dropzone
          formats={health?.supportedFormats ?? []}
          maxBytes={health?.limits.maxUploadBytes ?? 100 * 1024 * 1024}
          picked={picked}
          onPick={(p) => {
            setPicked(p);
            if (!title) setTitle(p.file.name.replace(/\.[^.]+$/, ""));
          }}
          onClear={() => setPicked(null)}
          disabled={uploading}
        />

        <AnimatePresence>
          {picked && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
              className="overflow-hidden"
            >
              <div className="pt-4 space-y-4">
                <label className="block">
                  <span className="text-sm text-muted">שם השיר</span>
                  <input
                    className="field mt-1.5"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    placeholder="איך לקרוא לשיר?"
                    disabled={uploading}
                  />
                </label>

                <div className="panel p-4 flex items-start gap-3">
                  <Mic size={18} className="text-muted mt-0.5 shrink-0" />
                  <div className="flex-1 min-w-0">
                    <label className="flex items-center gap-2.5 cursor-pointer">
                      <input
                        type="checkbox"
                        className="size-4 accent-[rgb(var(--glow))]"
                        checked={transcribe && canTranscribe}
                        disabled={!canTranscribe || uploading}
                        onChange={(e) => setTranscribe(e.target.checked)}
                      />
                      <span className="text-sm font-medium text-ink">
                        לתמלל גם את המילים
                      </span>
                    </label>
                    <p className="text-xs text-muted mt-1.5 leading-relaxed">
                      {canTranscribe ? (
                        <>Sends the audio to {settings?.transcriptionProvider} using your own API key.</>
                      ) : (
                        <>
                          Needs a transcription API key and your permission to send the
                          audio to that provider.{" "}
                          <button
                            type="button"
                            className="text-glow font-medium hover:underline"
                            onClick={onOpenSettings}
                          >
                            הגדרת שירות התמלול
                          </button>
                        </>
                      )}
                    </p>
                  </div>
                </div>

                <button
                  className="btn-primary w-full !py-3.5 text-base"
                  onClick={start}
                  disabled={uploading}
                >
                  {uploading ? (
                    <>
                      <Loader2 size={18} className="animate-spin" />
                      {uploadFraction < 1
                        ? `Uploading ${Math.round(uploadFraction * 100)}%`
                        : "Starting analysis"}
                    </>
                  ) : (
                    <>
                      נתח את השיר
                      <ArrowRight size={18} className="rtl:rotate-180" />
                    </>
                  )}
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.section>

      {recent.length > 0 && (
        <section className="pt-16">
          <h2 className="text-sm font-bold text-muted mb-4">Recent</h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <AnimatePresence mode="popLayout">
              {recent.map((p) => (
                <ProjectCard key={p.id} project={p} onDelete={remove} />
              ))}
            </AnimatePresence>
          </div>
        </section>
      )}

      {health && (
        <p className="pt-16 text-xs text-faint">
          Chord engine {health.engineVersion} — chords are derived from spectral
          analysis of your file, not looked up anywhere.
        </p>
      )}
    </div>
  );
}
