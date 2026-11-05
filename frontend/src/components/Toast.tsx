import { AnimatePresence, motion } from "framer-motion";
import { createContext, useCallback, useContext, useMemo, useState } from "react";
import { CheckCircle2, AlertTriangle, Info, X } from "lucide-react";

type Tone = "ok" | "error" | "info";
interface Toast { id: number; tone: Tone; title: string; body?: string }

const Ctx = createContext<{
  show: (tone: Tone, title: string, body?: string) => void;
}>({ show: () => {} });

export const useToast = () => useContext(Ctx);

const ICON = { ok: CheckCircle2, error: AlertTriangle, info: Info };
const TONE = {
  ok: "text-signal",
  error: "text-danger",
  info: "text-glow",
};

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [items, setItems] = useState<Toast[]>([]);

  const show = useCallback((tone: Tone, title: string, body?: string) => {
    const id = Date.now() + Math.random();
    setItems((xs) => [...xs, { id, tone, title, body }]);
    window.setTimeout(
      () => setItems((xs) => xs.filter((x) => x.id !== id)),
      tone === "error" ? 8000 : 4000,
    );
  }, []);

  const value = useMemo(() => ({ show }), [show]);

  return (
    <Ctx.Provider value={value}>
      {children}
      <div className="fixed bottom-4 inset-x-4 sm:inset-x-auto sm:end-6 sm:w-[min(26rem,calc(100vw-3rem))]
                      z-[80] flex flex-col gap-2 pointer-events-none">
        <AnimatePresence initial={false}>
          {items.map((t) => {
            const Icon = ICON[t.tone];
            return (
              <motion.div
                key={t.id}
                layout
                initial={{ opacity: 0, y: 12, scale: 0.97 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 8, scale: 0.97 }}
                transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
                className="panel shadow-deep px-4 py-3 flex gap-3 items-start pointer-events-auto"
              >
                <Icon size={18} className={`${TONE[t.tone]} shrink-0 mt-0.5`} />
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium text-ink">{t.title}</p>
                  {t.body && <p className="text-xs text-muted mt-1 break-words">{t.body}</p>}
                </div>
                <button
                  onClick={() => setItems((xs) => xs.filter((x) => x.id !== t.id))}
                  className="text-faint hover:text-ink transition-colors shrink-0"
                  aria-label="Dismiss"
                >
                  <X size={15} />
                </button>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </Ctx.Provider>
  );
}
