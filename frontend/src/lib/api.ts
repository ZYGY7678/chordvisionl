/**
 * Typed client for the ChordVision API.
 *
 * Every call goes through one function so that error handling, credentials
 * and JSON parsing are defined once. `credentials: "include"` matters: the
 * server identifies the browser with a signed cookie, and without it every
 * request would look like a different person and projects would vanish.
 */

const BASE = import.meta.env.VITE_API_BASE ?? "";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { credentials: "include", ...init });
  } catch {
    throw new ApiError("Can't reach the server. Check that it's running.", 0);
  }

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
      else if (Array.isArray(body?.detail) && body.detail[0]?.msg)
        detail = body.detail.map((d: any) => d.msg).join("; ");
    } catch {
      /* keep the generic message */
    }
    throw new ApiError(detail, res.status);
  }

  if (res.status === 204) return undefined as T;
  const text = await res.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

function json(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

// --- types -------------------------------------------------------------

export interface ChordAlternative { chordName: string; confidence: number }

export interface Chord {
  id: string;
  chordName: string;
  root: string;
  quality: string;
  bassNote: string | null;
  startTimeMs: number;
  endTimeMs: number;
  confidence: number;
  alternatives: ChordAlternative[];
  isEdited: boolean;
  detectionVersion: string;
}

export type TimingSource = "provider_word" | "forced_alignment" | "estimated" | "user";

export interface Word {
  id: string;
  text: string;
  startTimeMs: number;
  endTimeMs: number;
  confidence: number | null;
  lineId: number;
  wordIndex: number;
  timingSource: TimingSource;
  isEdited: boolean;
  language: string | null;
}

export interface ProjectSummary {
  id: string;
  title: string;
  fileName: string;
  durationMs: number;
  fileSize: number;
  status: "uploaded" | "analyzing" | "ready" | "failed";
  keyTonic: string | null;
  keyMode: string | null;
  tempoBpm: number | null;
  transposeSemitones: number;
  createdAt: string | null;
  hasLyrics: boolean;
  chordCount: number;
}

export interface Project extends ProjectSummary {
  keyConfidence: number | null;
  tuningOffsetCents: number | null;
  beatTimesMs: number[];
  onsetTimesMs: number[];
  waveformPeaks: [number, number][];
  analysisDiagnostics: Record<string, any>;
  chords: Chord[];
  words: Word[];
}

export interface PlacedChord {
  chordId: string;
  chordName: string;
  startTimeMs: number;
  endTimeMs: number;
  confidence: number;
  wordIndex: number | null;
  offsetRatio: number;
  anchor: "on_word" | "before_word" | "instrumental";
  isEdited: boolean;
}

export interface SongLine {
  lineId: number;
  startTimeMs: number;
  endTimeMs: number;
  isInstrumental: boolean;
  words: Word[];
  chords: PlacedChord[];
}

export interface SongView {
  projectId: string;
  title: string;
  key: string | null;
  keyConfidence: number | null;
  tempoBpm: number | null;
  transposeSemitones: number;
  durationMs: number;
  lines: SongLine[];
}

export interface Stage { id: string; label: string; measurable: boolean }

export interface Job {
  id: string;
  projectId: string;
  stage: string;
  progress: number;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  message: string | null;
  errorMessage: string | null;
  stages: Stage[];
}

export interface Health {
  status: string;
  version: string;
  engineVersion: string;
  stages: Stage[];
  limits: { maxUploadBytes: number; maxAudioSeconds: number };
  supportedFormats: string[];
}

export interface Provider {
  id: string;
  name: string;
  docsUrl: string;
  keyFormatHint: string;
  givesWordTimings: boolean;
  notes: string;
}

export interface Settings {
  transcriptionProvider: string | null;
  apiKeyHint: string | null;
  apiKeySet: boolean;
  apiKeyUpdatedAt: string | null;
  preferredLanguage: string | null;
  theme: "dark" | "light" | "system";
  chordNotation: "sharps" | "flats";
  allowAudioUpload: boolean;
}

// --- calls ---------------------------------------------------------------

export const api = {
  health: () => request<Health>("/api/health"),
  providers: () => request<Provider[]>("/api/providers"),

  getSettings: () => request<Settings>("/api/settings"),
  saveSettings: (patch: Partial<Settings> & { apiKey?: string }) =>
    request<Settings>("/api/settings", json("PUT", patch)),
  deleteApiKey: () => request<{ ok: boolean }>("/api/settings/api-key", { method: "DELETE" }),
  verifyKey: (provider: string, apiKey: string) =>
    request<{ valid: boolean; message: string }>(
      "/api/settings/verify-key", json("POST", { provider, apiKey })),

  listProjects: () => request<ProjectSummary[]>("/api/projects"),
  getProject: (id: string) => request<Project>(`/api/projects/${id}`),
  patchProject: (id: string, patch: { title?: string; transposeSemitones?: number }) =>
    request<Project>(`/api/projects/${id}`, json("PATCH", patch)),
  deleteProject: (id: string) =>
    request<{ ok: boolean }>(`/api/projects/${id}`, { method: "DELETE" }),

  upload: (file: File, title: string, transcribe: boolean, onProgress?: (f: number) => void) =>
    new Promise<{ project: ProjectSummary; jobId: string }>((resolve, reject) => {
      // XHR rather than fetch: fetch still cannot report upload progress,
      // and a long upload with no feedback is the worst part of any tool
      // like this.
      const form = new FormData();
      form.append("file", file);
      form.append("title", title);
      form.append("transcribe", String(transcribe));

      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${BASE}/api/projects`);
      xhr.withCredentials = true;
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText));
        } else {
          let detail = `Upload failed (${xhr.status})`;
          try { detail = JSON.parse(xhr.responseText).detail ?? detail; } catch { /* */ }
          reject(new ApiError(detail, xhr.status));
        }
      };
      xhr.onerror = () => reject(new ApiError("Can't reach the server.", 0));
      xhr.send(form);
    }),

  job: (projectId: string) => request<Job>(`/api/projects/${projectId}/job`),
  reanalyze: (projectId: string, transcribe: boolean) =>
    request<{ jobId: string }>(`/api/projects/${projectId}/analyze`, json("POST", { transcribe })),
  song: (projectId: string) => request<SongView>(`/api/projects/${projectId}/song`),

  audioUrl: (projectId: string) => `${BASE}/api/projects/${projectId}/audio`,
  exportUrl: (projectId: string, fmt: string) =>
    `${BASE}/api/projects/${projectId}/export/${fmt}`,

  addChord: (projectId: string, body: { chordName: string; startTimeMs: number; endTimeMs: number }) =>
    request<Chord>(`/api/projects/${projectId}/chords`, json("POST", body)),
  patchChord: (projectId: string, chordId: string,
               body: { chordName?: string; startTimeMs?: number; endTimeMs?: number }) =>
    request<Chord>(`/api/projects/${projectId}/chords/${chordId}`, json("PATCH", body)),
  deleteChord: (projectId: string, chordId: string) =>
    request<{ ok: boolean }>(`/api/projects/${projectId}/chords/${chordId}`, { method: "DELETE" }),

  addWord: (projectId: string, body: { text: string; startTimeMs: number; endTimeMs: number; lineId: number }) =>
    request<Word>(`/api/projects/${projectId}/words`, json("POST", body)),
  patchWord: (projectId: string, wordId: string,
              body: { text?: string; startTimeMs?: number; endTimeMs?: number; lineId?: number }) =>
    request<Word>(`/api/projects/${projectId}/words/${wordId}`, json("PATCH", body)),
  splitWord: (projectId: string, wordId: string, atRatio: number) =>
    request<Word[]>(`/api/projects/${projectId}/words/${wordId}/split`, json("POST", { atRatio })),
  deleteWord: (projectId: string, wordId: string) =>
    request<{ ok: boolean }>(`/api/projects/${projectId}/words/${wordId}`, { method: "DELETE" }),
};
