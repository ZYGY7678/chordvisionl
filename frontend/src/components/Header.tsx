import { Link, useLocation } from "react-router-dom";
import { Moon, Sun, Settings as SettingsIcon, Library } from "lucide-react";
import { Logo } from "./Logo";
import { useTheme } from "../hooks/useTheme";

export function Header({ onOpenSettings }: { onOpenSettings: () => void }) {
  const { resolved, toggle } = useTheme();
  const { pathname } = useLocation();

  return (
    <header className="sticky top-0 z-40 border-b border-line bg-base/85 backdrop-blur-xl">
      <div className="mx-auto max-w-6xl px-5 h-16 flex items-center gap-3">
        <Link to="/" className="flex items-center gap-2.5 me-auto group">
          <Logo />
          <span className="text-[17px] font-bold tracking-tight text-ink">
            ChordVision
          </span>
        </Link>

        <Link
          to="/projects"
          className={`btn-ghost ${pathname === "/projects" ? "text-ink bg-raised" : ""}`}
        >
          <Library size={17} />
          <span className="hidden sm:inline">השירים שלי</span>
        </Link>

        <button className="btn-ghost" onClick={onOpenSettings}>
          <SettingsIcon size={17} />
          <span className="hidden sm:inline">הגדרות</span>
        </button>

        <button
          className="btn-ghost !px-2.5"
          onClick={toggle}
          aria-label={resolved === "dark" ? "מעבר למצב בהיר" : "מעבר למצב כהה"}
          title={resolved === "dark" ? "Switch to light theme" : "Switch to dark theme"}
        >
          {resolved === "dark" ? <Sun size={17} /> : <Moon size={17} />}
        </button>
      </div>
    </header>
  );
}
