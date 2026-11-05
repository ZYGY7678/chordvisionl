import { useCallback, useEffect, useState } from "react";

export type Theme = "dark" | "light" | "system";

const KEY = "chordvision.theme";

function systemPrefersDark(): boolean {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? true;
}

function apply(theme: Theme) {
  const resolved = theme === "system" ? (systemPrefersDark() ? "dark" : "light") : theme;
  document.documentElement.setAttribute("data-theme", resolved);
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(
    () => (localStorage.getItem(KEY) as Theme | null) ?? "dark",
  );

  useEffect(() => {
    apply(theme);
    localStorage.setItem(KEY, theme);
    if (theme !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => apply("system");
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [theme]);

  const setTheme = useCallback((t: Theme) => setThemeState(t), []);
  const toggle = useCallback(() => {
    setThemeState((t) => {
      const resolved = t === "system" ? (systemPrefersDark() ? "dark" : "light") : t;
      return resolved === "dark" ? "light" : "dark";
    });
  }, []);

  const resolved: "dark" | "light" =
    theme === "system" ? (systemPrefersDark() ? "dark" : "light") : theme;

  return { theme, resolved, setTheme, toggle };
}
