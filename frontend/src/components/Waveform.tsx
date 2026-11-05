import { useCallback, useMemo, useRef, useState } from "react";

/**
 * The waveform, drawn on a canvas from the peak pairs the server computed.
 *
 * Canvas rather than SVG: a four minute song gives around 1200 peak pairs,
 * and 2400 SVG nodes that re-render on every animation frame is enough to
 * make the playhead stutter.
 */
export function Waveform({
  peaks,
  positionMs,
  durationMs,
  loop,
  beatsMs,
  onSeek,
  onSelectRegion,
  height = 56,
}: {
  peaks: [number, number][];
  positionMs: number;
  durationMs: number;
  loop: { startMs: number; endMs: number } | null;
  beatsMs?: number[];
  onSeek: (ms: number) => void;
  onSelectRegion?: (region: { startMs: number; endMs: number } | null) => void;
  height?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const boxRef = useRef<HTMLDivElement | null>(null);
  const [dragFrom, setDragFrom] = useState<number | null>(null);
  const [dragTo, setDragTo] = useState<number | null>(null);

  const progress = durationMs > 0 ? positionMs / durationMs : 0;

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const box = boxRef.current;
    if (!canvas || !box) return;

    const dpr = window.devicePixelRatio || 1;
    const w = box.clientWidth;
    const h = height;
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const styles = getComputedStyle(document.documentElement);
    const faint = `rgb(${styles.getPropertyValue("--faint").trim()})`;
    const glow = `rgb(${styles.getPropertyValue("--glow").trim()})`;

    const mid = h / 2;
    const count = peaks.length || 1;
    const barW = Math.max(1, w / count);
    const playedX = progress * w;

    for (let i = 0; i < count; i++) {
      const [lo, hi] = peaks[i] ?? [0, 0];
      const x = (i / count) * w;
      const top = mid - Math.abs(hi) * (h / 2) * 0.94;
      const bottom = mid + Math.abs(lo) * (h / 2) * 0.94;
      ctx.fillStyle = x < playedX ? glow : faint;
      ctx.globalAlpha = x < playedX ? 0.95 : 0.4;
      ctx.fillRect(x, top, Math.max(barW - 0.5, 0.7), Math.max(bottom - top, 1));
    }
    ctx.globalAlpha = 1;
  }, [peaks, progress, height]);

  useMemo(() => {
    // Redraw on the next frame whenever inputs change.
    requestAnimationFrame(draw);
    return null;
  }, [draw]);

  const msAt = (clientX: number) => {
    const box = boxRef.current;
    if (!box || durationMs <= 0) return 0;
    const rect = box.getBoundingClientRect();
    const ratio = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1);
    return ratio * durationMs;
  };

  const pct = (ms: number) => (durationMs > 0 ? (ms / durationMs) * 100 : 0);

  const selStart = dragFrom !== null && dragTo !== null ? Math.min(dragFrom, dragTo) : null;
  const selEnd = dragFrom !== null && dragTo !== null ? Math.max(dragFrom, dragTo) : null;

  return (
    <div
      ref={boxRef}
      className="relative w-full cursor-pointer select-none touch-none"
      style={{ height }}
      onPointerDown={(e) => {
        (e.target as Element).setPointerCapture?.(e.pointerId);
        if (e.shiftKey && onSelectRegion) {
          setDragFrom(msAt(e.clientX));
          setDragTo(msAt(e.clientX));
        } else {
          onSeek(msAt(e.clientX));
        }
      }}
      onPointerMove={(e) => {
        if (dragFrom !== null) setDragTo(msAt(e.clientX));
      }}
      onPointerUp={() => {
        if (dragFrom !== null && dragTo !== null && onSelectRegion) {
          const a = Math.min(dragFrom, dragTo);
          const b = Math.max(dragFrom, dragTo);
          onSelectRegion(b - a > 400 ? { startMs: a, endMs: b } : null);
        }
        setDragFrom(null);
        setDragTo(null);
      }}
      role="slider"
      aria-label="Seek"
      aria-valuemin={0}
      aria-valuemax={Math.round(durationMs)}
      aria-valuenow={Math.round(positionMs)}
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "ArrowRight") onSeek(positionMs + 5000);
        if (e.key === "ArrowLeft") onSeek(positionMs - 5000);
      }}
    >
      {beatsMs && beatsMs.length > 0 && durationMs > 0 && (
        <div className="absolute inset-0 pointer-events-none opacity-30">
          {beatsMs.map((b, i) =>
            i % 4 === 0 ? (
              <span
                key={i}
                className="absolute top-0 bottom-0 w-px bg-faint"
                style={{ left: `${pct(b)}%` }}
              />
            ) : null,
          )}
        </div>
      )}

      <canvas ref={canvasRef} className="block w-full" />

      {loop && (
        <div
          className="absolute inset-y-0 bg-signal/15 border-x border-signal/60 pointer-events-none"
          style={{ left: `${pct(loop.startMs)}%`, width: `${pct(loop.endMs - loop.startMs)}%` }}
        />
      )}

      {selStart !== null && selEnd !== null && (
        <div
          className="absolute inset-y-0 bg-signal/25 pointer-events-none"
          style={{ left: `${pct(selStart)}%`, width: `${pct(selEnd - selStart)}%` }}
        />
      )}

      <div
        className="absolute top-0 bottom-0 w-[2px] bg-ink pointer-events-none"
        style={{ left: `${progress * 100}%` }}
      />
    </div>
  );
}
