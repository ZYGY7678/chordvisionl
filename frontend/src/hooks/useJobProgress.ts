import { useEffect, useRef, useState } from "react";
import { api, type Job, type Stage } from "../lib/api";

export interface Progress {
  stage: string;
  progress: number;
  status: Job["status"];
  message: string | null;
  errorMessage: string | null;
  stages: Stage[];
}

/**
 * Follow an analysis job.
 *
 * Server-sent events are the primary channel. If the stream cannot be
 * opened - a proxy that buffers, a network that drops it - the hook falls
 * back to polling rather than leaving the user looking at a frozen bar.
 * The fallback is silent because from the user's side nothing differs
 * except the update rate.
 */
export function useJobProgress(projectId: string | null, enabled: boolean) {
  const [progress, setProgress] = useState<Progress | null>(null);
  const doneRef = useRef(false);

  useEffect(() => {
    if (!projectId || !enabled) return;
    doneRef.current = false;

    let source: EventSource | null = null;
    let pollTimer: number | null = null;
    let cancelled = false;

    const finish = (p: Progress) => {
      setProgress(p);
      if (p.status === "succeeded" || p.status === "failed") doneRef.current = true;
    };

    const poll = async () => {
      if (cancelled || doneRef.current) return;
      try {
        const job = await api.job(projectId);
        finish({
          stage: job.stage, progress: job.progress, status: job.status,
          message: job.message, errorMessage: job.errorMessage,
          stages: job.stages ?? [],
        });
      } catch {
        /* keep trying: the worker may simply not have written a row yet */
      }
      if (!cancelled && !doneRef.current) pollTimer = window.setTimeout(poll, 700);
    };

    try {
      source = new EventSource(`/api/projects/${projectId}/events`, { withCredentials: true });
      let stages: Stage[] = [];

      source.addEventListener("stages", (e) => {
        try { stages = JSON.parse((e as MessageEvent).data); } catch { /* */ }
      });

      const onProgress = (e: Event) => {
        try {
          const d = JSON.parse((e as MessageEvent).data);
          finish({ ...d, stages });
        } catch { /* */ }
      };
      source.addEventListener("progress", onProgress);
      source.addEventListener("done", (e) => {
        onProgress(e);
        source?.close();
      });
      source.onerror = () => {
        source?.close();
        source = null;
        if (!doneRef.current && !cancelled) poll();
      };
    } catch {
      poll();
    }

    return () => {
      cancelled = true;
      source?.close();
      if (pollTimer) window.clearTimeout(pollTimer);
    };
  }, [projectId, enabled]);

  return progress;
}
