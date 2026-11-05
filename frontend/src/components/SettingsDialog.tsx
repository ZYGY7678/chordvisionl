import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Eye, EyeOff, ShieldCheck, Loader2, Trash2, ExternalLink, Lock } from "lucide-react";
import { api, ApiError, type Provider, type Settings } from "../lib/api";
import { useToast } from "./Toast";
import { useTheme, type Theme } from "../hooks/useTheme";

export function SettingsDialog({
  open,
  settings,
  onClose,
  onSaved,
}: {
  open: boolean;
  settings: Settings | null;
  onClose: () => void;
  onSaved: (s: Settings) => void;
}) {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [provider, setProvider] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [language, setLanguage] = useState("");
  const [notation, setNotation] = useState<"sharps" | "flats">("sharps");
  const [allowUpload, setAllowUpload] = useState(false);
  const [checking, setChecking] = useState(false);
  const [saving, setSaving] = useState(false);
  const { theme, setTheme } = useTheme();
  const toast = useToast();

  useEffect(() => {
    if (!open) return;
    api.providers().then(setProviders).catch(() => {});
  }, [open]);

  useEffect(() => {
    if (!settings) return;
    setProvider(settings.transcriptionProvider ?? "");
    setLanguage(settings.preferredLanguage ?? "");
    setNotation(settings.chordNotation);
    setAllowUpload(settings.allowAudioUpload);
    setApiKey("");
  }, [settings, open]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    if (open) document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const selected = providers.find((p) => p.id === provider);

  async function verify() {
    if (!provider || !apiKey) return;
    setChecking(true);
    try {
      const res = await api.verifyKey(provider, apiKey);
      toast.show(res.valid ? "ok" : "error",
        res.valid ? "Key verified" : "Key rejected", res.message);
    } catch (err) {
      toast.show("error", "Could not check the key",
        err instanceof ApiError ? err.message : String(err));
    } finally {
      setChecking(false);
    }
  }

  async function save() {
    setSaving(true);
    try {
      const next = await api.saveSettings({
        transcriptionProvider: provider || null,
        ...(apiKey ? { apiKey } : {}),
        preferredLanguage: language || null,
        chordNotation: notation,
        allowAudioUpload: allowUpload,
      });
      onSaved(next);
      setApiKey("");
      toast.show("ok", "Settings saved");
      onClose();
    } catch (err) {
      toast.show("error", "Could not save settings",
        err instanceof ApiError ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function removeKey() {
    try {
      await api.deleteApiKey();
      const next = await api.getSettings();
      onSaved(next);
      toast.show("ok", "API key removed");
    } catch (err) {
      toast.show("error", "Could not remove the key",
        err instanceof ApiError ? err.message : String(err));
    }
  }

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[60]"
            onClick={onClose}
          />
          <motion.div
            role="dialog" aria-modal="true" aria-label="Settings"
            initial={{ opacity: 0, y: 16, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.985 }}
            transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
            className="fixed inset-x-4 top-[6vh] bottom-[6vh] sm:inset-x-auto sm:start-1/2
                       sm:-translate-x-1/2 sm:w-[min(34rem,calc(100vw-2rem))]
                       panel shadow-deep z-[61] flex flex-col overflow-hidden"
          >
            <div className="flex items-center gap-3 px-5 py-4 border-b border-line">
              <h2 className="text-lg font-bold text-ink me-auto">Settings</h2>
              <button className="btn-ghost !px-2" onClick={onClose} aria-label="Close settings">
                <X size={18} />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto px-5 py-5 space-y-7">
              <section>
                <h3 className="text-sm font-bold text-ink mb-1">Appearance</h3>
                <p className="text-xs text-muted mb-3">
                  Dark is the default; the song view was designed in it.
                </p>
                <div className="flex gap-1.5">
                  {(["dark", "light", "system"] as Theme[]).map((t) => (
                    <button
                      key={t}
                      onClick={() => setTheme(t)}
                      className={`flex-1 py-2 rounded-xl text-sm capitalize transition-colors border
                                  ${theme === t
                                    ? "bg-raised border-glow/50 text-ink font-bold"
                                    : "bg-base border-line text-muted hover:text-ink"}`}
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </section>

              <section>
                <h3 className="text-sm font-bold text-ink mb-1">Chord spelling</h3>
                <p className="text-xs text-muted mb-3">
                  Whether a black key reads as A# or B♭ everywhere in the app.
                </p>
                <div className="flex gap-1.5">
                  {(["sharps", "flats"] as const).map((n) => (
                    <button
                      key={n}
                      onClick={() => setNotation(n)}
                      className={`flex-1 py-2 rounded-xl text-sm transition-colors border
                                  ${notation === n
                                    ? "bg-raised border-glow/50 text-ink font-bold"
                                    : "bg-base border-line text-muted hover:text-ink"}`}
                    >
                      {n === "sharps" ? "Sharps (A#)" : "Flats (B♭)"}
                    </button>
                  ))}
                </div>
              </section>

              <section>
                <h3 className="text-sm font-bold text-ink mb-1">Lyric transcription</h3>
                <p className="text-xs text-muted mb-3 leading-relaxed">
                  Chord detection runs entirely on this server. Transcription does
                  not: it needs an outside service and an API key you supply.
                </p>

                <label className="block mb-3">
                  <span className="text-xs text-muted">Service</span>
                  <select
                    className="field mt-1"
                    value={provider}
                    onChange={(e) => setProvider(e.target.value)}
                  >
                    <option value="">None — chords only</option>
                    {providers.map((p) => (
                      <option key={p.id} value={p.id}>{p.name}</option>
                    ))}
                  </select>
                </label>

                {selected && (
                  <>
                    <p className="text-xs text-muted mb-3 leading-relaxed">
                      {selected.notes}
                      {selected.docsUrl && (
                        <>
                          {" "}
                          <a href={selected.docsUrl} target="_blank" rel="noreferrer"
                             className="text-glow hover:underline inline-flex items-center gap-0.5">
                            Where to get a key <ExternalLink size={11} />
                          </a>
                        </>
                      )}
                    </p>

                    <label className="block">
                      <span className="text-xs text-muted">
                        API key {settings?.apiKeySet && (
                          <span className="text-faint">
                            — one is saved ({settings.apiKeyHint}); typing replaces it
                          </span>
                        )}
                      </span>
                      <div className="relative mt-1">
                        <input
                          className="field pe-24"
                          type={showKey ? "text" : "password"}
                          value={apiKey}
                          onChange={(e) => setApiKey(e.target.value)}
                          placeholder={selected.keyFormatHint}
                          autoComplete="off"
                          spellCheck={false}
                        />
                        <div className="absolute inset-y-0 end-1.5 flex items-center gap-0.5">
                          <button className="btn-ghost !px-2 !py-1.5"
                                  onClick={() => setShowKey((s) => !s)}
                                  aria-label={showKey ? "Hide the key" : "Show the key"}>
                            {showKey ? <EyeOff size={15} /> : <Eye size={15} />}
                          </button>
                          <button className="btn-ghost !px-2 !py-1.5"
                                  onClick={verify} disabled={!apiKey || checking}
                                  title="Check this key works">
                            {checking
                              ? <Loader2 size={15} className="animate-spin" />
                              : <ShieldCheck size={15} />}
                          </button>
                        </div>
                      </div>
                    </label>

                    <p className="text-xs text-faint mt-2 flex items-start gap-1.5 leading-relaxed">
                      <Lock size={12} className="mt-0.5 shrink-0" />
                      The key is encrypted on the server. It is never sent back to
                      this page, never written to a log, and never stored in your
                      browser.
                    </p>

                    {settings?.apiKeySet && (
                      <button className="btn-danger !px-2 !py-1.5 mt-2 text-xs" onClick={removeKey}>
                        <Trash2 size={13} />
                        Remove the saved key
                      </button>
                    )}

                    <label className="block mt-4">
                      <span className="text-xs text-muted">Lyric language</span>
                      <input
                        className="field mt-1"
                        value={language}
                        onChange={(e) => setLanguage(e.target.value)}
                        placeholder="auto-detect — or he, en, es…"
                      />
                    </label>

                    <label className="flex items-start gap-2.5 mt-4 cursor-pointer panel p-3">
                      <input
                        type="checkbox"
                        className="size-4 accent-[rgb(var(--glow))] mt-0.5"
                        checked={allowUpload}
                        onChange={(e) => setAllowUpload(e.target.checked)}
                      />
                      <span>
                        <span className="block text-sm text-ink font-medium">
                          Allow sending audio to {selected.name}
                        </span>
                        <span className="block text-xs text-muted mt-0.5 leading-relaxed">
                          Required for transcription. Your recording leaves this
                          server and goes to them. Chord detection never does.
                        </span>
                      </span>
                    </label>
                  </>
                )}
              </section>
            </div>

            <div className="px-5 py-4 border-t border-line flex gap-2">
              <button className="btn-ghost" onClick={onClose}>Cancel</button>
              <button className="btn-primary flex-1" onClick={save} disabled={saving}>
                {saving && <Loader2 size={15} className="animate-spin" />}
                Save settings
              </button>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
