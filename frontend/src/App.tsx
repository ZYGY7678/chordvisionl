import { useEffect, useState } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { Header } from "./components/Header";
import { SettingsDialog } from "./components/SettingsDialog";
import { ToastProvider } from "./components/Toast";
import { Home } from "./pages/Home";
import { Projects } from "./pages/Projects";
import { Song } from "./pages/Song";
import { api, type Health, type Settings } from "./lib/api";
import { useTheme } from "./hooks/useTheme";

function Shell() {
  const [health, setHealth] = useState<Health | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  useTheme();

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    api.getSettings().then(setSettings).catch(() => setSettings(null));
  }, []);

  return (
    <>
      <Header onOpenSettings={() => setSettingsOpen(true)} />
      <Routes>
        <Route
          path="/"
          element={
            <Home health={health} settings={settings}
                  onOpenSettings={() => setSettingsOpen(true)} />
          }
        />
        <Route path="/projects" element={<Projects />} />
        <Route path="/song/:id" element={<Song settings={settings} />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <SettingsDialog
        open={settingsOpen}
        settings={settings}
        onClose={() => setSettingsOpen(false)}
        onSaved={setSettings}
      />
    </>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <ToastProvider>
        <Shell />
      </ToastProvider>
    </BrowserRouter>
  );
}
