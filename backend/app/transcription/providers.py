"""Concrete transcription adapters.

Honest status note
------------------
These adapters are written against each provider's documented HTTP API
and request exactly the fields the pipeline needs, in particular
word-level timestamps.  They have **not** been executed against the live
services, because that requires paid API keys that this build does not
have.  Treat them as reviewed-but-unverified: the request shapes, the
response parsing and the error handling are all real code, but the first
person to plug in a real key should expect to check them.

The ``file`` provider below *is* exercised end to end, and is what the
automated tests and the offline demo path use.
"""
from __future__ import annotations

import json
import os
from typing import Optional, List, Dict, Any

import httpx

from .base import (
    TranscriptionProvider, TranscriptionResult, TranscriptionError,
    Word, Segment, ProviderInfo,
)
from ..db import TimingSource


class OpenAIWhisperProvider(TranscriptionProvider):
    """OpenAI's audio transcription endpoint.

    ``timestamp_granularities[]=word`` is what makes it return a start and
    end for each word rather than only for each segment.
    """

    info = ProviderInfo(
        id="openai",
        name="OpenAI Whisper",
        docs_url="https://platform.openai.com/docs/guides/speech-to-text",
        key_format_hint="starts with sk-",
        gives_word_timings=True,
        notes="Good multilingual coverage, including Hebrew.",
    )

    BASE = "https://api.openai.com/v1"
    MODEL = "whisper-1"

    async def verify_key(self) -> bool:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(
                f"{self.BASE}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        if r.status_code == 200:
            return True
        if r.status_code in (401, 403):
            return False
        raise TranscriptionError(
            f"OpenAI returned HTTP {r.status_code} while checking the key.",
            status_code=r.status_code, retryable=r.status_code >= 500,
        )

    async def transcribe(self, audio_path: str,
                         language: Optional[str] = None) -> TranscriptionResult:
        data = {
            "model": self.MODEL,
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word", "segment"],
        }
        if language:
            data["language"] = language

        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                with open(audio_path, "rb") as fh:
                    r = await client.post(
                        f"{self.BASE}/audio/transcriptions",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        data=data,
                        files={"file": (os.path.basename(audio_path), fh,
                                        "application/octet-stream")},
                    )
        except httpx.HTTPError as exc:
            raise TranscriptionError(
                f"Could not reach OpenAI: {self._safe_error(exc)}", retryable=True
            ) from exc

        if r.status_code != 200:
            raise TranscriptionError(
                f"OpenAI transcription failed (HTTP {r.status_code}): "
                f"{self._safe_error(Exception(r.text[:300]))}",
                status_code=r.status_code, retryable=r.status_code >= 500,
            )

        payload = r.json()
        words: List[Word] = []
        for w in payload.get("words") or []:
            words.append(Word(
                text=str(w.get("word", "")).strip(),
                start_ms=self._ms(w.get("start")),
                end_ms=self._ms(w.get("end")),
                timing_source=TimingSource.PROVIDER_WORD.value,
            ))

        segments: List[Segment] = []
        for s in payload.get("segments") or []:
            segments.append(Segment(
                text=str(s.get("text", "")).strip(),
                start_ms=self._ms(s.get("start")),
                end_ms=self._ms(s.get("end")),
                confidence=None,
            ))

        return TranscriptionResult(
            provider=self.info.id,
            language=payload.get("language"),
            text=payload.get("text", ""),
            segments=segments,
            words=words,
            has_word_timings=bool(words),
            raw=None,
        )


class DeepgramProvider(TranscriptionProvider):
    """Deepgram's prerecorded endpoint."""

    info = ProviderInfo(
        id="deepgram",
        name="Deepgram",
        docs_url="https://developers.deepgram.com/docs/pre-recorded-audio",
        key_format_hint="a 40-character token",
        gives_word_timings=True,
        notes="Fast, and returns a confidence value for every word.",
    )

    BASE = "https://api.deepgram.com/v1"

    async def verify_key(self) -> bool:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(
                f"{self.BASE}/projects",
                headers={"Authorization": f"Token {self._api_key}"},
            )
        if r.status_code == 200:
            return True
        if r.status_code in (401, 403):
            return False
        raise TranscriptionError(
            f"Deepgram returned HTTP {r.status_code} while checking the key.",
            status_code=r.status_code, retryable=r.status_code >= 500,
        )

    async def transcribe(self, audio_path: str,
                         language: Optional[str] = None) -> TranscriptionResult:
        params: Dict[str, Any] = {
            "model": "nova-2", "smart_format": "true",
            "punctuate": "true", "utterances": "true",
        }
        if language:
            params["language"] = language

        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                with open(audio_path, "rb") as fh:
                    r = await client.post(
                        f"{self.BASE}/listen",
                        headers={"Authorization": f"Token {self._api_key}",
                                 "Content-Type": "application/octet-stream"},
                        params=params, content=fh.read(),
                    )
        except httpx.HTTPError as exc:
            raise TranscriptionError(
                f"Could not reach Deepgram: {self._safe_error(exc)}", retryable=True
            ) from exc

        if r.status_code != 200:
            raise TranscriptionError(
                f"Deepgram transcription failed (HTTP {r.status_code}): "
                f"{self._safe_error(Exception(r.text[:300]))}",
                status_code=r.status_code, retryable=r.status_code >= 500,
            )

        payload = r.json()
        try:
            channel = payload["results"]["channels"][0]
            alt = channel["alternatives"][0]
        except (KeyError, IndexError) as exc:
            raise TranscriptionError("Deepgram returned no transcript.") from exc

        words = [
            Word(
                text=str(w.get("punctuated_word") or w.get("word", "")),
                start_ms=self._ms(w.get("start")),
                end_ms=self._ms(w.get("end")),
                confidence=w.get("confidence"),
                timing_source=TimingSource.PROVIDER_WORD.value,
            )
            for w in alt.get("words") or []
        ]

        segments: List[Segment] = []
        for utt in payload.get("results", {}).get("utterances") or []:
            segments.append(Segment(
                text=str(utt.get("transcript", "")),
                start_ms=self._ms(utt.get("start")),
                end_ms=self._ms(utt.get("end")),
                confidence=utt.get("confidence"),
            ))

        return TranscriptionResult(
            provider=self.info.id,
            language=(payload.get("results", {}).get("channels", [{}])[0]
                      .get("detected_language") or language),
            text=alt.get("transcript", ""),
            segments=segments,
            words=words,
            has_word_timings=bool(words),
        )


class AssemblyAIProvider(TranscriptionProvider):
    """AssemblyAI: upload, start a job, poll until it finishes."""

    info = ProviderInfo(
        id="assemblyai",
        name="AssemblyAI",
        docs_url="https://www.assemblyai.com/docs",
        key_format_hint="a 32-character hex string",
        gives_word_timings=True,
        notes="Asynchronous: the adapter uploads, then polls for the result.",
    )

    BASE = "https://api.assemblyai.com/v2"

    async def verify_key(self) -> bool:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"{self.BASE}/transcript",
                                 headers={"authorization": self._api_key},
                                 params={"limit": 1})
        if r.status_code == 200:
            return True
        if r.status_code in (401, 403):
            return False
        raise TranscriptionError(
            f"AssemblyAI returned HTTP {r.status_code} while checking the key.",
            status_code=r.status_code, retryable=r.status_code >= 500,
        )

    async def transcribe(self, audio_path: str,
                         language: Optional[str] = None) -> TranscriptionResult:
        import asyncio
        headers = {"authorization": self._api_key}

        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                with open(audio_path, "rb") as fh:
                    up = await client.post(f"{self.BASE}/upload",
                                           headers=headers, content=fh.read())
                if up.status_code != 200:
                    raise TranscriptionError(
                        f"AssemblyAI upload failed (HTTP {up.status_code}).",
                        status_code=up.status_code)
                upload_url = up.json()["upload_url"]

                body: Dict[str, Any] = {"audio_url": upload_url, "punctuate": True}
                if language:
                    body["language_code"] = language
                else:
                    body["language_detection"] = True

                start = await client.post(f"{self.BASE}/transcript",
                                          headers=headers, json=body)
                if start.status_code not in (200, 201):
                    raise TranscriptionError(
                        f"AssemblyAI job could not be started "
                        f"(HTTP {start.status_code}).", status_code=start.status_code)
                job_id = start.json()["id"]

                deadline = self._timeout_s
                waited = 0.0
                while waited < deadline:
                    await asyncio.sleep(3.0)
                    waited += 3.0
                    poll = await client.get(f"{self.BASE}/transcript/{job_id}",
                                            headers=headers)
                    data = poll.json()
                    status = data.get("status")
                    if status == "completed":
                        break
                    if status == "error":
                        raise TranscriptionError(
                            f"AssemblyAI reported an error: "
                            f"{self._safe_error(Exception(str(data.get('error'))[:300]))}")
                else:
                    raise TranscriptionError(
                        "AssemblyAI did not finish within the timeout.", retryable=True)
        except httpx.HTTPError as exc:
            raise TranscriptionError(
                f"Could not reach AssemblyAI: {self._safe_error(exc)}", retryable=True
            ) from exc

        words = [
            Word(text=str(w.get("text", "")),
                 start_ms=int(w.get("start") or 0),
                 end_ms=int(w.get("end") or 0),
                 confidence=w.get("confidence"),
                 timing_source=TimingSource.PROVIDER_WORD.value)
            for w in data.get("words") or []
        ]
        return TranscriptionResult(
            provider=self.info.id,
            language=data.get("language_code"),
            text=data.get("text", ""),
            segments=[],
            words=words,
            has_word_timings=bool(words),
        )


class GeminiProvider(TranscriptionProvider):
    """Google Gemini 3.5 Transcribe, through the Interactions API.

    This one has a free tier, so it is the provider to try first: a key from
    https://aistudio.google.com/apikey costs nothing.  The flow is

      1. shrink the recording with ffmpeg (mono, 16 kHz, small MP3), which
         keeps the upload tiny and avoids every size limit;
      2. upload it to the Files API and wait until it is ACTIVE;
      3. ask ``gemini-3.5-transcribe`` for a verbatim transcript with
         word-level timestamps;
      4. delete the uploaded file again.

    Word timestamps arrive as ``word_info`` annotations with offsets such as
    ``"1.250s"``.  Google documents two spellings of the audio input in its
    REST examples, so the request is sent in the SDK spelling first and, if
    the server answers 400, once more in the other one.
    """

    info = ProviderInfo(
        id="gemini",
        name="Google Gemini (free)",
        docs_url="https://ai.google.dev/gemini-api/docs/transcribe",
        key_format_hint="starts with AIza - free key from aistudio.google.com/apikey",
        gives_word_timings=True,
        notes="Has a free tier. Word timings, Hebrew included. "
              "The free tier has a small daily request limit.",
    )

    BASE = "https://generativelanguage.googleapis.com"
    MODEL = "gemini-3.5-transcribe"

    # The app stores short codes ("he"); Gemini wants BCP-47 tags.
    _LANG = {
        "he": "he-IL", "iw": "he-IL", "en": "en-US", "es": "es-419", "fr": "fr-FR",
        "de": "de-DE", "it": "it-IT", "pt": "pt-BR", "ru": "ru-RU", "ar": "ar-EG",
        "tr": "tr-TR", "ja": "ja-JP", "ko": "ko-KR", "zh": "cmn-Hans-CN",
        "hi": "hi-IN", "nl": "nl-NL", "pl": "pl-PL", "uk": "uk-UA", "fa": "fa-IR",
    }

    def _headers(self, **extra: str) -> Dict[str, str]:
        h = {"x-goog-api-key": self._api_key}
        h.update(extra)
        return h

    async def verify_key(self) -> bool:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"{self.BASE}/v1beta/models",
                                 headers=self._headers(), params={"pageSize": 1})
        if r.status_code == 200:
            return True
        # Google answers an unknown key with 400 (API_KEY_INVALID), not 401.
        if r.status_code in (400, 401, 403):
            return False
        raise TranscriptionError(
            f"Gemini returned HTTP {r.status_code} while checking the key.",
            status_code=r.status_code, retryable=r.status_code >= 500,
        )

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _shrink(audio_path: str):
        """Return (path_to_upload, mime_type, temp_file_or_None)."""
        import shutil
        import subprocess
        import tempfile

        if shutil.which("ffmpeg"):
            fd, tmp = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-v", "error", "-i", audio_path, "-vn",
                     "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "48k", tmp],
                    check=True, capture_output=True, timeout=300,
                )
                if os.path.getsize(tmp) > 0:
                    return tmp, "audio/mp3", tmp
            except (subprocess.SubprocessError, OSError):
                pass
            try:
                os.remove(tmp)
            except OSError:
                pass

        ext = os.path.splitext(audio_path)[1].lower()
        mime = {".mp3": "audio/mp3", ".wav": "audio/wav", ".flac": "audio/flac",
                ".ogg": "audio/ogg", ".m4a": "audio/m4a", ".aac": "audio/aac",
                ".opus": "audio/opus", ".webm": "audio/webm"}.get(ext)
        if mime is None:
            raise TranscriptionError("Could not prepare this audio format for Gemini.")
        return audio_path, mime, None

    async def _upload(self, client: httpx.AsyncClient, path: str, mime: str) -> Dict[str, Any]:
        size = os.path.getsize(path)
        start = await client.post(
            f"{self.BASE}/upload/v1beta/files",
            headers=self._headers(**{
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(size),
                "X-Goog-Upload-Header-Content-Type": mime,
                "Content-Type": "application/json",
            }),
            json={"file": {"display_name": "chordvision-audio"}},
        )
        self._check(start, "upload start")
        upload_url = start.headers.get("x-goog-upload-url")
        if not upload_url:
            raise TranscriptionError("Gemini did not return an upload address.")

        with open(path, "rb") as fh:
            data = fh.read()
        fin = await client.post(
            upload_url,
            headers={"Content-Length": str(size), "X-Goog-Upload-Offset": "0",
                     "X-Goog-Upload-Command": "upload, finalize"},
            content=data,
        )
        self._check(fin, "upload")
        info = fin.json().get("file") or {}
        if not info.get("uri"):
            raise TranscriptionError("Gemini accepted the upload but returned no file address.")
        return info

    async def _wait_active(self, client: httpx.AsyncClient, info: Dict[str, Any]) -> Dict[str, Any]:
        import asyncio
        name = info.get("name")
        waited = 0.0
        while info.get("state", "ACTIVE") == "PROCESSING" and name and waited < 120:
            await asyncio.sleep(2.0)
            waited += 2.0
            r = await client.get(f"{self.BASE}/v1beta/{name}", headers=self._headers())
            self._check(r, "file status")
            info = r.json()
        if info.get("state") == "FAILED":
            raise TranscriptionError("Gemini could not process the audio file.")
        return info

    def _check(self, r: httpx.Response, what: str) -> None:
        if r.status_code < 300:
            return
        if r.status_code == 429:
            raise TranscriptionError(
                "Gemini's free request limit was reached. Wait a minute and try "
                "again, or try again tomorrow if the daily limit is used up.",
                status_code=429, retryable=True)
        if r.status_code in (400, 401, 403) and what != "interaction":
            raise TranscriptionError(
                f"Gemini rejected the request ({what}, HTTP {r.status_code}). "
                "Check that the API key is correct.", status_code=r.status_code)
        raise TranscriptionError(
            f"Gemini {what} failed (HTTP {r.status_code}): "
            f"{self._safe_error(Exception(r.text[:300]))}",
            status_code=r.status_code, retryable=r.status_code >= 500)

    @staticmethod
    def _seconds(v: Any) -> float:
        if v is None:
            return 0.0
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().rstrip("sS")
        try:
            return float(s)
        except ValueError:
            return 0.0

    def _config(self, language: Optional[str]) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "mode": {"type": "verbatim", "timestamp_granularities": ["word"]},
        }
        if language:
            code = language if "-" in language else self._LANG.get(language.lower())
            if code:
                cfg["language_codes"] = [code]
        return {"transcription_config": cfg}

    # -- main ------------------------------------------------------------

    async def transcribe(self, audio_path: str,
                         language: Optional[str] = None) -> TranscriptionResult:
        upload_path, mime, tmp = self._shrink(audio_path)
        file_name = None
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                try:
                    info = await self._upload(client, upload_path, mime)
                    file_name = info.get("name")
                    info = await self._wait_active(client, info)
                    uri = info["uri"]
                    file_mime = info.get("mimeType") or mime

                    attempts = [
                        {"model": self.MODEL,
                         "input": [{"type": "audio", "uri": uri, "mime_type": file_mime}],
                         "generation_config": self._config(language)},
                        {"model": f"models/{self.MODEL}",
                         "input": [{"type": "file", "file_uri": uri}],
                         "generation_config": self._config(language)},
                    ]
                    r = None
                    for body in attempts:
                        r = await client.post(f"{self.BASE}/v1beta/interactions",
                                              headers=self._headers(), json=body)
                        if r.status_code != 400:
                            break
                    self._check(r, "interaction")
                    payload = r.json()
                finally:
                    if file_name:
                        try:
                            await client.delete(f"{self.BASE}/v1beta/{file_name}",
                                                headers=self._headers())
                        except httpx.HTTPError:
                            pass
        except httpx.HTTPError as exc:
            raise TranscriptionError(
                f"Could not reach Gemini: {self._safe_error(exc)}", retryable=True
            ) from exc
        finally:
            if tmp:
                try:
                    os.remove(tmp)
                except OSError:
                    pass

        return self._parse(payload, language)

    def _parse(self, payload: Dict[str, Any], language: Optional[str]) -> TranscriptionResult:
        if payload.get("status") not in (None, "completed"):
            raise TranscriptionError(
                f"Gemini did not finish the transcription (status {payload.get('status')}).")
        words: List[Word] = []
        texts: List[str] = []
        for step in payload.get("steps") or []:
            if step.get("type") not in (None, "model_output"):
                continue
            for content in step.get("content") or []:
                if content.get("text"):
                    texts.append(str(content["text"]))
                for ann in content.get("annotations") or []:
                    if ann.get("type") != "word_info":
                        continue
                    text = str(ann.get("text", "")).strip()
                    if not text:
                        continue
                    words.append(Word(
                        text=text,
                        start_ms=self._ms(self._seconds(ann.get("start_offset"))),
                        end_ms=self._ms(self._seconds(ann.get("end_offset"))),
                        timing_source=TimingSource.PROVIDER_WORD.value,
                    ))
        full = " ".join(texts).strip() or " ".join(w.text for w in words)
        if not words:
            if full:
                raise TranscriptionError(
                    "Gemini returned a transcript without word timings, so the "
                    "words cannot be placed under the chords. Try again.",
                    retryable=True)
            raise TranscriptionError(
                "Gemini found no speech in this recording.")
        return TranscriptionResult(
            provider=self.info.id,
            language=language,
            text=full,
            segments=[],
            words=words,
            has_word_timings=True,
        )


class FileProvider(TranscriptionProvider):
    """Reads a transcript from a local JSON file instead of calling a service.

    This is not a toy.  It covers three real needs:

      * the automated tests need a deterministic transcript with no
        network and no key;
      * a user who already has a transcript (from a DAW, a lyrics file, a
        previous run) can bring it rather than paying to generate it
        again;
      * it is the only provider that can be used when the recording must
        not leave the machine.

    It also doubles as the reference for what an adapter has to return,
    including the segment-only case that forces the aligner to run.

    Expected JSON: ``{"language": "...", "text": "...", "words":[...]}``
    or ``{"segments":[{"text":..., "start":..., "end":...}]}`` with times
    in seconds (or ``startTimeMs``/``endTimeMs`` in milliseconds).
    """

    info = ProviderInfo(
        id="file",
        name="Local transcript file",
        docs_url="",
        key_format_hint="the absolute path to a .json transcript",
        gives_word_timings=False,
        notes="Nothing is uploaded anywhere. Use when the audio must stay local.",
    )

    async def verify_key(self) -> bool:
        return os.path.isfile(self._api_key)

    async def transcribe(self, audio_path: str,
                         language: Optional[str] = None) -> TranscriptionResult:
        path = self._api_key
        if not os.path.isfile(path):
            raise TranscriptionError(f"No transcript file at {path}")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise TranscriptionError(f"Could not read the transcript file: {exc}") from exc

        def _time_ms(entry: Dict[str, Any], *keys: str) -> int:
            for k in keys:
                if k in entry and entry[k] is not None:
                    v = entry[k]
                    return int(v) if k.endswith("Ms") else int(round(float(v) * 1000))
            return 0

        words = [
            Word(text=str(w.get("text") or w.get("word") or ""),
                 start_ms=_time_ms(w, "startTimeMs", "start"),
                 end_ms=_time_ms(w, "endTimeMs", "end"),
                 confidence=w.get("confidence"),
                 timing_source=TimingSource.PROVIDER_WORD.value)
            for w in payload.get("words") or []
        ]
        segments = [
            Segment(text=str(s.get("text", "")),
                    start_ms=_time_ms(s, "startTimeMs", "start"),
                    end_ms=_time_ms(s, "endTimeMs", "end"),
                    confidence=s.get("confidence"))
            for s in payload.get("segments") or []
        ]
        return TranscriptionResult(
            provider=self.info.id,
            language=payload.get("language") or language,
            text=payload.get("text", " ".join(w.text for w in words)),
            segments=segments,
            words=words,
            has_word_timings=bool(words),
        )


PROVIDERS: Dict[str, type] = {
    GeminiProvider.info.id: GeminiProvider,
    OpenAIWhisperProvider.info.id: OpenAIWhisperProvider,
    DeepgramProvider.info.id: DeepgramProvider,
    AssemblyAIProvider.info.id: AssemblyAIProvider,
    FileProvider.info.id: FileProvider,
}


def get_provider(provider_id: str, api_key: str,
                 timeout_s: float = 600.0) -> TranscriptionProvider:
    cls = PROVIDERS.get(provider_id)
    if cls is None:
        raise TranscriptionError(
            f"Unknown transcription provider {provider_id!r}. "
            f"Available: {', '.join(sorted(PROVIDERS))}"
        )
    return cls(api_key=api_key, timeout_s=timeout_s)


def list_providers() -> List[Dict[str, Any]]:
    return [
        {
            "id": cls.info.id, "name": cls.info.name,
            "docsUrl": cls.info.docs_url,
            "keyFormatHint": cls.info.key_format_hint,
            "givesWordTimings": cls.info.gives_word_timings,
            "notes": cls.info.notes,
        }
        for cls in PROVIDERS.values()
    ]
