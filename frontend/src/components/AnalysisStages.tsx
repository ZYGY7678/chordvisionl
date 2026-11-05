import { motion } from "framer-motion";
import { Check, Loader2, Circle, AlertTriangle } from "lucide-react";
import type { Stage } from "../lib/api";
import type { Progress } from "../hooks/useJobProgress";

/**
 * The stage list during analysis.
 *
 * Two rules from the brief are visible here. A stage shows a percentage
 * only when that percentage is measured - the transcription stage is
 * marked unmeasurable because it is one call to a third party that
 * reports nothing until it returns, so it shows motion instead of a
 * number. And the overall bar is the worker's real progress, never a
 * timer pretending to be one.
 */
export function AnalysisStages({
  stages,
  progress,
}: {
  stages: Stage[];
  progress: Progress | null;
}) {
  const list = progress?.stages?.length ? progress.stages : stages;
  const currentIndex = list.findIndex((s) => s.id === progress?.stage);
  const failed = progress?.status === "failed";

  return (
    <ol className="space-y-1">
      {list.map((stage, i) => {
        const state =
          failed && i === currentIndex
            ? "failed"
            : currentIndex < 0
            ? "pending"
            : i < currentIndex || progress?.status === "succeeded"
            ? "done"
            : i === currentIndex
            ? "running"
            : "pending";

        return (
          <li key={stage.id}>
            <motion.div
              layout
              className={`flex items-center gap-3 rounded-xl px-3 py-2.5 transition-colors
                          ${state === "running" ? "bg-raised" : ""}`}
            >
              <span className="shrink-0 size-5 grid place-items-center">
                {state === "done" && <Check size={16} className="text-signal" />}
                {state === "running" && <Loader2 size={16} className="text-glow animate-spin" />}
                {state === "failed" && <AlertTriangle size={16} className="text-danger" />}
                {state === "pending" && <Circle size={9} className="text-faint" />}
              </span>

              <span
                className={`text-sm flex-1 ${
                  state === "pending" ? "text-faint" : state === "failed" ? "text-danger" : "text-ink"
                }`}
              >
                {stage.label}
              </span>

              {state === "running" && (
                <span className="text-xs text-muted truncate max-w-[14rem] text-end">
                  {stage.measurable ? progress?.message : "working…"}
                </span>
              )}
            </motion.div>
          </li>
        );
      })}
    </ol>
  );
}
