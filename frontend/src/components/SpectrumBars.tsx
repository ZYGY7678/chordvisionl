import { motion, useReducedMotion } from "framer-motion";

/**
 * Twelve bars for the twelve pitch classes.
 *
 * This is the one piece of motion on the home screen, and it only moves
 * while something is actually happening - a file being dragged over the
 * drop area, or an analysis running. Idle, it holds still. Ambient
 * animation on a page that is waiting for the user is noise.
 */
export function SpectrumBars({
  active = false,
  height = 48,
  bars = 12,
  className = "",
}: {
  active?: boolean;
  height?: number;
  bars?: number;
  className?: string;
}) {
  const reduce = useReducedMotion();
  // A fixed, non-random pattern: it reads as a chord held steady rather
  // than as a random equaliser.
  const resting = [0.3, 0.55, 0.22, 0.7, 0.38, 0.9, 0.3, 0.62, 0.26, 0.78, 0.34, 0.5];

  return (
    <div
      className={`flex items-end gap-[3px] ${className}`}
      style={{ height }}
      aria-hidden="true"
    >
      {Array.from({ length: bars }).map((_, i) => {
        const base = resting[i % resting.length];
        return (
          <motion.span
            key={i}
            // h-full matters: these are flex children scaled on the Y
            // axis, and without an intrinsic height there is nothing to
            // scale, so the bars render at zero and the box looks empty.
            className="flex-1 h-full rounded-full bg-glow/70"
            style={{ transformOrigin: "bottom" }}
            initial={false}
            animate={
              active && !reduce
                ? { scaleY: [base, Math.min(1, base + 0.45), base * 0.75, base] }
                : { scaleY: base }
            }
            transition={
              active && !reduce
                ? { duration: 1.1 + (i % 5) * 0.17, repeat: Infinity, ease: "easeInOut" }
                : { duration: 0.35, ease: [0.22, 1, 0.36, 1] }
            }
          />
        );
      })}
    </div>
  );
}
