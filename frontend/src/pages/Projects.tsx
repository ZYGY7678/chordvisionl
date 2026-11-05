import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { AnimatePresence } from "framer-motion";
import { Plus } from "lucide-react";
import { api, ApiError, type ProjectSummary } from "../lib/api";
import { ProjectCard } from "../components/ProjectCard";
import { useToast } from "../components/Toast";

export function Projects() {
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const toast = useToast();

  useEffect(() => {
    api.listProjects().then(setProjects).catch(() => setProjects([]));
  }, []);

  async function remove(id: string) {
    try {
      await api.deleteProject(id);
      setProjects((ps) => (ps ?? []).filter((p) => p.id !== id));
      toast.show("ok", "השיר נמחק");
    } catch (err) {
      toast.show("error", "לא ניתן למחוק את השיר",
        err instanceof ApiError ? err.message : String(err));
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-5 py-14">
      <div className="flex items-end gap-4 mb-8">
        <h1 className="text-3xl font-black text-ink me-auto">השירים שלי</h1>
        <Link to="/" className="btn-primary">
          <Plus size={17} />
          שיר חדש
        </Link>
      </div>

      {projects === null && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <div key={i} className="panel p-5 h-40 animate-pulse" />
          ))}
        </div>
      )}

      {projects?.length === 0 && (
        <div className="panel p-12 text-center">
          <p className="text-ink font-medium">עדיין אין כאן שירים</p>
          <p className="text-sm text-muted mt-1.5">
            מעלים שיר והוא יופיע כאן עם האקורדים שלו.
          </p>
          <Link to="/" className="btn-primary mt-6">העלאת שיר</Link>
        </div>
      )}

      {projects && projects.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <AnimatePresence mode="popLayout">
            {projects.map((p) => (
              <ProjectCard key={p.id} project={p} onDelete={remove} />
            ))}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}
