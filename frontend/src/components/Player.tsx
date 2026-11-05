import { useEffect } from "react";
import {
  Play, Pause, Square, RotateCcw, RotateCw, Volume2, VolumeX, Repeat, X,
} from "lucide-react";
import { Waveform } from "./Waveform";
import { formatTime } from "../lib/format";
import type { usePlayer } from "../hooks/usePlayer";

const SPEEDS = [0.5, 0.75, 1, 1.25, 1.5];

export function Player({
  player,
  peaks,
  beatsMs,
}: {
  player: ReturnType<typeof usePlayer>;
  peaks: [number, number][];
  beatsMs: number[];
}) {
  // Keyboard control. Space is the one shortcut every player has, and a
  // musician working through a song has their hands on an instrument,
  // not on a mouse.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      if (target?.isContentEditable) return;

      if (e.code === "Space") {
        e.preventDefault();
        player.toggle();
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        player.nudge(-5000);
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        player.nudge(5000);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [player]);

  return (
    <div className="border-t border-line bg-surface/95 backdrop-blur-xl">
      <div className="mx-auto max-w-5xl px-5 py-3">
        <Waveform
          peaks={peaks}
          positionMs={player.positionMs}
          durationMs={player.durationMs}
          loop={player.loop}
          beatsMs={beatsMs}
          onSeek={player.seek}
          onSelectRegion={player.setLoop}
          height={52}
        />

        <div className="mt-2.5 flex items-center gap-1.5 flex-wrap">
          <button
            className="btn-primary !px-3.5 !py-2.5 !rounded-full"
            onClick={player.toggle}
            disabled={!player.ready}
            aria-label={player.playing ? "Pause" : "Play"}
          >
            {player.playing ? <Pause size={18} /> : <Play size={18} className="translate-x-px" />}
          </button>

          <button className="btn-ghost !px-2.5" onClick={player.stop}
                  disabled={!player.ready} aria-label="Stop">
            <Square size={16} />
          </button>
          <button className="btn-ghost !px-2.5" onClick={() => player.nudge(-5000)}
                  disabled={!player.ready} aria-label="Back 5 seconds" title="Back 5 seconds">
            <RotateCcw size={16} />
          </button>
          <button className="btn-ghost !px-2.5" onClick={() => player.nudge(5000)}
                  disabled={!player.ready} aria-label="Forward 5 seconds" title="Forward 5 seconds">
            <RotateCw size={16} />
          </button>

          <span className="ms-2 text-sm text-muted tabular-nums">
            {formatTime(player.positionMs)}
            <span className="text-faint"> / {formatTime(player.durationMs)}</span>
          </span>

          <div className="ms-auto flex items-center gap-1.5 flex-wrap">
            {player.loop ? (
              <button
                className="chip !text-signal !border-signal/50 hover:!bg-signal/10"
                onClick={() => player.setLoop(null)}
              >
                <Repeat size={12} />
                Loop {formatTime(player.loop.startMs)}–{formatTime(player.loop.endMs)}
                <X size={12} />
              </button>
            ) : (
              <span className="hidden md:inline text-xs text-faint">
                shift-drag the waveform to loop a section
              </span>
            )}

            <div className="flex items-center rounded-xl bg-base border border-line p-0.5">
              {SPEEDS.map((s) => (
                <button
                  key={s}
                  onClick={() => player.setSpeed(s)}
                  className={`px-2 py-1 text-xs rounded-lg transition-colors tabular-nums
                              ${player.rate === s
                                ? "bg-raised text-ink font-bold"
                                : "text-muted hover:text-ink"}`}
                >
                  {s}x
                </button>
              ))}
            </div>

            <button className="btn-ghost !px-2.5" onClick={player.toggleMute}
                    aria-label={player.muted ? "Unmute" : "Mute"}>
              {player.muted ? <VolumeX size={16} /> : <Volume2 size={16} />}
            </button>
            <input
              type="range"
              min={0} max={1} step={0.01}
              value={player.muted ? 0 : player.volume}
              onChange={(e) => player.setVolume(Number(e.target.value))}
              className="w-20 accent-[rgb(var(--glow))]"
              aria-label="Volume"
            />
          </div>
        </div>

        {player.error && (
          <p className="mt-2 text-xs text-danger">{player.error}</p>
        )}
      </div>
    </div>
  );
}
