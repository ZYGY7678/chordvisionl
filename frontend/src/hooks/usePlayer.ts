import { useCallback, useEffect, useRef, useState } from "react";

export interface LoopRegion { startMs: number; endMs: number }

/**
 * Audio playback, with the element's own clock as the source of truth.
 *
 * The brief is specific that synchronisation must follow the player's
 * `currentTime` rather than a timer, and the reason is worth stating: a
 * `setInterval` drifts away from the audio within seconds, and a drifting
 * highlight is worse than none, because it tells the player to change
 * chord at the wrong moment.
 *
 * The position is therefore sampled on every animation frame straight
 * from the element. That also means the highlight stays correct through
 * seeking, buffering, tab throttling and speed changes, none of which a
 * timer survives.
 */
export function usePlayer(src: string | null) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const loopRef = useRef<LoopRegion | null>(null);

  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [positionMs, setPositionMs] = useState(0);
  const [durationMs, setDurationMs] = useState(0);
  const [rate, setRate] = useState(1);
  const [volume, setVolumeState] = useState(1);
  const [muted, setMuted] = useState(false);
  const [loop, setLoopState] = useState<LoopRegion | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loopRef.current = loop;
  }, [loop]);

  useEffect(() => {
    if (!src) return;
    const audio = new Audio();
    audio.preload = "metadata";
    audio.crossOrigin = "use-credentials";
    audio.src = src;
    audioRef.current = audio;
    setReady(false);
    setError(null);

    const onMeta = () => {
      setDurationMs(Number.isFinite(audio.duration) ? audio.duration * 1000 : 0);
      setReady(true);
    };
    const onEnd = () => setPlaying(false);
    const onErr = () =>
      setError("The audio could not be loaded. Try reloading the page.");

    audio.addEventListener("loadedmetadata", onMeta);
    audio.addEventListener("ended", onEnd);
    audio.addEventListener("error", onErr);

    return () => {
      audio.pause();
      audio.removeEventListener("loadedmetadata", onMeta);
      audio.removeEventListener("ended", onEnd);
      audio.removeEventListener("error", onErr);
      audio.src = "";
      audioRef.current = null;
    };
  }, [src]);

  // One rAF loop while playing. Reading the element each frame is what
  // keeps the highlight locked to the sound.
  useEffect(() => {
    if (!playing) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      return;
    }
    const tick = () => {
      const audio = audioRef.current;
      if (audio) {
        const ms = audio.currentTime * 1000;
        const region = loopRef.current;
        if (region && ms >= region.endMs) {
          audio.currentTime = region.startMs / 1000;
          setPositionMs(region.startMs);
        } else {
          setPositionMs(ms);
        }
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [playing]);

  const play = useCallback(async () => {
    const audio = audioRef.current;
    if (!audio) return;
    try {
      await audio.play();
      setPlaying(true);
    } catch {
      setError("The browser blocked playback. Press play again.");
    }
  }, []);

  const pause = useCallback(() => {
    audioRef.current?.pause();
    setPlaying(false);
  }, []);

  const toggle = useCallback(() => {
    if (playing) pause();
    else void play();
  }, [playing, pause, play]);

  const seek = useCallback((ms: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    const clamped = Math.max(0, Math.min(ms, durationMs || ms));
    audio.currentTime = clamped / 1000;
    setPositionMs(clamped);
  }, [durationMs]);

  const stop = useCallback(() => {
    pause();
    seek(0);
  }, [pause, seek]);

  const nudge = useCallback((deltaMs: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    seek(audio.currentTime * 1000 + deltaMs);
  }, [seek]);

  const setSpeed = useCallback((value: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    // preservesPitch keeps a slowed-down song in the same key, which is
    // the entire point of slowing it down to learn a part.
    (audio as any).preservesPitch = true;
    (audio as any).mozPreservesPitch = true;
    (audio as any).webkitPreservesPitch = true;
    audio.playbackRate = value;
    setRate(value);
  }, []);

  const setVolume = useCallback((value: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.volume = value;
    setVolumeState(value);
    if (value > 0 && audio.muted) {
      audio.muted = false;
      setMuted(false);
    }
  }, []);

  const toggleMute = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.muted = !audio.muted;
    setMuted(audio.muted);
  }, []);

  const setLoop = useCallback((region: LoopRegion | null) => setLoopState(region), []);

  return {
    ready, playing, positionMs, durationMs, rate, volume, muted, loop, error,
    play, pause, toggle, stop, seek, nudge, setSpeed, setVolume, toggleMute, setLoop,
  };
}
