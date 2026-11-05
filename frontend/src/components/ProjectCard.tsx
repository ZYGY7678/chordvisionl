import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Trash2, AlertTriangle, Loader2 } from "lucide-react";
import type { ProjectSummary } from "../lib/api";
import { formatTime, formatDate } from "../lib/format";

/** A tiny static waveform drawn from the project id.
 *  It is a visual fingerprint so the cards are distinguishable at a
 *  glance - it is explicitly not the song's real waveform, which is why
 *  it is drawn flat and dim rather than presented as data. */
function Fingerprint({ seed }: { seed: string }) {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0;
  const bars = Array.from({ length: 28 }, (_, i) => {
    h = (h * 1664525 + 1013904223) >>> 0;
    return 0.2 + ((h >>> 16) % 80) / 100;
  });
  return (
    <div className="flex items-end gap-[2px] h-8" aria-hidden="true">
      {bars.map((v, i) => (
        <span key={i} className="flex-1 rounded-full bg-faint/35" style={{ height: `${v * 100}%` }} />
      ))}
    </div>
  );
}

export function ProjectCard({
  project,
  onDelete,
}: {
  project: ProjectSummary;
  onDelete: (id: string) => void;
}) {
  const busy = project.status === "analyzing" || project.status === "uploaded";
  const failed = project.status === "failed";

  return (
    <motion.div layout className="panel p-5 group relative hover:border-faint transition-colors">
      <Link to={`/song/${project.id}`} className="block">
        <Fingerprint seed={project.id} />
        <h3 className="mt-4 font-bold text-ink truncate pe-8">{project.title}</h3>
        <p className="text-sm text-muted mt-1">
          {formatTime(project.durationMs)}
          {project.keyTonic ? `  ·  ${project.keyTonic} ${project.keyMode}` : ""}
          {project.tempoBpm ? `  ·  ${Math.round(project.tempoBpm)} BPM` : ""}
        </p>
        <div className="mt-3.5 flex items-center gap-2 flex-wrap">
          {busy && (
            <span className="chip !text-glow !border-glow/40">
              <Loader2 size={12} className="animate-spin" />
              Analyzing
            </span>
          )}
          {failed && (
            <span className="chip !text-danger !border-danger/40">
              <AlertTriangle size={12} />
              Failed
            </span>
          )}
          {!busy && !failed && (
            <>
              <span className="chip">{project.chordCount} chords</span>
              {project.hasLyrics && <span className="chip">lyrics</span>}
            </>
          )}
          <span className="text-xs text-faint ms-auto">{formatDate(project.createdAt)}</span>
        </div>
      </Link>

      <button
        onClick={(e) => {
          e.preventDefault();
          onDelete(project.id);
        }}
        className="absolute top-4 end-4 p-2 rounded-lg text-faint opacity-0 group-hover:opacity-100
                   focus-visible:opacity-100 hover:text-danger hover:bg-danger/10 transition"
        aria-label={`Delete ${project.title}`}
      >
        <Trash2 size={15} />
      </button>
    </motion.div>
  );
}
