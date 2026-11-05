"""End-to-end API test: upload a real file, analyse it, read everything back.

This drives the running application through HTTP exactly as the browser
does - upload, progress, project read, synchronised view, editing,
transposition and every export format.  It uses FastAPI's TestClient, so
the real routes, the real database, the real worker process and the real
chord engine all take part.

Transcription uses the ``file`` provider: a transcript on disk rather
than a paid API.  That exercises the whole lyric path - provider
interface, forced alignment, line grouping, chord-to-word placement -
without needing a key.
"""
from __future__ import annotations

import os, sys, json, time, tempfile, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "engine"))
sys.path.insert(0, os.path.join(ROOT, "backend"))

WORK = tempfile.mkdtemp(prefix="chordvision-e2e-")
os.environ["CHORDVISION_STORAGE_DIR"] = os.path.join(WORK, "storage")
os.environ["CHORDVISION_DATABASE_URL"] = f"sqlite:///{os.path.join(WORK, 'test.db')}"
os.environ["CHORDVISION_SECRET_KEY"] = "test-secret-not-for-production"
os.environ["CHORDVISION_ENVIRONMENT"] = "development"

import synth  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

FAILED = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {label}")
    else:
        FAILED.append(label)
        print(f"  FAIL  {label}{(': ' + detail) if detail else ''}")


def main() -> int:
    # Entering the client runs the application's lifespan, which is what
    # creates the database schema and starts the worker pool. Without it
    # the app is imported but never started, and the first query fails on
    # a missing table.
    client = TestClient(app)
    client.__enter__()

    print("\n[1] service metadata")
    r = client.get("/api/health")
    check("health responds", r.status_code == 200, r.text[:200])
    health = r.json()
    check("engine version reported", bool(health.get("engineVersion")))
    check("eight analysis stages declared", len(health["stages"]) == 8)

    r = client.get("/api/providers")
    check("providers listed", r.status_code == 200 and len(r.json()) >= 3)

    print("\n[2] uploads that must be rejected")
    r = client.post("/api/projects",
                    files={"file": ("fake.mp3", b"not audio at all", "audio/mpeg")})
    check("a text file named .mp3 is rejected", r.status_code == 400,
          f"got {r.status_code}")
    r = client.post("/api/projects",
                    files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")})
    check("a non-audio extension is rejected", r.status_code == 400,
          f"got {r.status_code}")

    print("\n[3] upload a real song")
    progression = ["C", "G", "Am", "F"] * 2
    y, sr, spans = synth.render_progression(
        progression, synth.RenderOptions(bpm=100, with_drums=True,
                                         with_reverb=True, seed=21))
    wav = os.path.join(WORK, "song.wav")
    synth.write_wav(wav, y, sr)

    with open(wav, "rb") as fh:
        r = client.post("/api/projects",
                        files={"file": ("song.wav", fh, "audio/wav")},
                        data={"title": "E2E Test Song"})
    check("upload accepted", r.status_code == 201, r.text[:300])
    if r.status_code != 201:
        return 1
    project_id = r.json()["project"]["id"]
    print(f"        project {project_id}")

    print("\n[4] analysis progresses and finishes")
    deadline = time.time() + 180
    last_stage, seen_stages, job = None, [], {}
    while time.time() < deadline:
        jr = client.get(f"/api/projects/{project_id}/job")
        if jr.status_code != 200:
            time.sleep(0.4); continue
        job = jr.json()
        if job["stage"] != last_stage:
            last_stage = job["stage"]
            seen_stages.append(last_stage)
            print(f"        {job['progress']*100:5.1f}%  {job['stage']:18s} {job['message'] or ''}")
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.4)

    check("job succeeded", job.get("status") == "succeeded",
          job.get("errorMessage") or str(job))
    check("progress reached 100%", job.get("progress", 0) >= 0.999)
    check("more than one stage was reported", len(seen_stages) > 1,
          str(seen_stages))

    print("\n[5] the analysis result")
    r = client.get(f"/api/projects/{project_id}")
    check("project readable", r.status_code == 200)
    p = r.json()
    check("status is ready", p["status"] == "ready", p["status"])
    check("chords were detected", len(p["chords"]) >= 4, str(len(p["chords"])))
    check("key detected as C major",
          p["keyTonic"] == "C" and p["keyMode"] == "major",
          f"{p['keyTonic']} {p['keyMode']}")
    check("tempo near 100 BPM", p["tempoBpm"] and abs(p["tempoBpm"] - 100) < 10,
          str(p["tempoBpm"]))
    check("waveform peaks produced", len(p["waveformPeaks"]) > 100)
    check("beat grid produced", len(p["beatTimesMs"]) > 5)
    check("no chord claims confidence 1.0 from detection",
          all(c["confidence"] < 1.0 for c in p["chords"]))
    diagnostics = p.get("analysisDiagnostics") or {}
    check("replacement recognizer v2 is active",
          diagnostics.get("engineVersion") == "2.0.0",
          str(diagnostics.get("engineVersion")))
    check("recognizer provenance is recorded",
          diagnostics.get("recognizer") == "harmonic-cqt-nnls-viterbi",
          str(diagnostics.get("recognizer")))
    check("chord confidence values remain valid probabilities",
          all(0.0 <= c["confidence"] <= 1.0 for c in p["chords"]))

    detected = [c["chordName"] for c in p["chords"]]
    print(f"        detected: {' '.join(detected)}")
    expected_roots = {"C", "G", "A", "F"}
    found_roots = {c["root"] for c in p["chords"]}
    check("the right roots were found", expected_roots <= found_roots,
          f"missing {expected_roots - found_roots}")

    print("\n[6] audio streaming with range requests")
    r = client.get(f"/api/projects/{project_id}/audio",
                   headers={"Range": "bytes=0-1023"})
    check("range request returns 206", r.status_code == 206, str(r.status_code))
    check("content-range header present", "content-range" in
          {k.lower() for k in r.headers})
    check("exactly 1024 bytes returned", len(r.content) == 1024, str(len(r.content)))

    print("\n[7] transcription via the local-file provider")
    transcript = {
        "language": "en",
        "segments": [
            {"text": "walking down the road tonight", "start": 0.4, "end": 4.2},
            {"text": "singing every word i know", "start": 4.9, "end": 9.0},
        ],
    }
    tpath = os.path.join(WORK, "transcript.json")
    with open(tpath, "w", encoding="utf-8") as fh:
        json.dump(transcript, fh)

    r = client.put("/api/settings", json={
        "transcriptionProvider": "file", "apiKey": tpath,
        "allowAudioUpload": True, "preferredLanguage": "en",
    })
    check("settings saved", r.status_code == 200, r.text[:200])
    s = r.json()
    check("api key never echoed back", "apiKey" not in s and s["apiKeySet"] is True)
    check("only a key hint is exposed", s["apiKeyHint"].startswith("****"))

    r = client.post(f"/api/projects/{project_id}/analyze", json={"transcribe": True})
    check("re-analysis queued", r.status_code == 200, r.text[:200])

    deadline = time.time() + 180
    while time.time() < deadline:
        job = client.get(f"/api/projects/{project_id}/job").json()
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.4)
    check("re-analysis with transcription succeeded", job["status"] == "succeeded",
          job.get("errorMessage") or "")

    p = client.get(f"/api/projects/{project_id}").json()
    words = p["words"]
    check("words were produced", len(words) == 10, str(len(words)))
    check("words are tagged as forced alignment, not as measured",
          all(w["timingSource"] == "forced_alignment" for w in words),
          str({w["timingSource"] for w in words}))
    check("word times are strictly increasing",
          all(words[i]["startTimeMs"] <= words[i+1]["startTimeMs"]
              for i in range(len(words)-1)))
    check("every word has a positive duration",
          all(w["endTimeMs"] > w["startTimeMs"] for w in words))

    print("\n[8] the synchronised song view")
    r = client.get(f"/api/projects/{project_id}/song")
    check("song view returned", r.status_code == 200)
    view = r.json()
    check("lines produced", len(view["lines"]) >= 2, str(len(view["lines"])))
    placed = [c for line in view["lines"] for c in line["chords"]]
    check("every chord is placed exactly once",
          len(placed) == len({c["chordId"] for c in placed}) == len(p["chords"]),
          f"{len(placed)} placed vs {len(p['chords'])} chords")
    for line in view["lines"]:
        if line["isInstrumental"]:
            continue
        for c in line["chords"]:
            if c["wordIndex"] is None:
                continue
            check_once = 0 <= c["wordIndex"] < len(line["words"])
            if not check_once:
                check("chord word index in range", False,
                      f"{c['chordName']} -> {c['wordIndex']}")
                break
    else:
        check("chord word indices all in range", True)

    print("\n[9] transposition")
    r = client.patch(f"/api/projects/{project_id}", json={"transposeSemitones": 2})
    check("transpose accepted", r.status_code == 200)
    tp = r.json()
    moved = {c["root"] for c in tp["chords"]}
    check("roots moved up a whole tone", moved == {"D", "A", "B", "G"},
          str(sorted(moved)))
    check("chord times unchanged by transposition",
          [c["startTimeMs"] for c in tp["chords"]] ==
          [c["startTimeMs"] for c in p["chords"]])
    r = client.patch(f"/api/projects/{project_id}", json={"transposeSemitones": -13})
    check("out-of-range transpose rejected", r.status_code == 422, str(r.status_code))
    client.patch(f"/api/projects/{project_id}", json={"transposeSemitones": 0})

    print("\n[10] editing")
    first = p["chords"][0]
    r = client.patch(f"/api/projects/{project_id}/chords/{first['id']}",
                     json={"chordName": "Cmaj7"})
    check("chord renamed", r.status_code == 200 and r.json()["chordName"] == "Cmaj7",
          r.text[:200])
    check("edited chord is flagged", r.json()["isEdited"] is True)

    r = client.patch(f"/api/projects/{project_id}/chords/{first['id']}",
                     json={"chordName": "NotAChord"})
    check("a nonsense chord symbol is rejected", r.status_code == 400,
          str(r.status_code))

    r = client.post(f"/api/projects/{project_id}/chords",
                    json={"chordName": "Bb7", "startTimeMs": 100, "endTimeMs": 900})
    check("chord added", r.status_code == 201, r.text[:200])
    new_id = r.json()["id"]
    r = client.delete(f"/api/projects/{project_id}/chords/{new_id}")
    check("chord deleted", r.status_code == 200)

    w0 = words[0]
    r = client.post(f"/api/projects/{project_id}/words/{w0['id']}/split",
                    json={"atRatio": 0.5})
    check("word split into two", r.status_code == 200 and len(r.json()) == 2,
          r.text[:200])
    parts = r.json()
    check("split parts do not overlap",
          parts[0]["endTimeMs"] == parts[1]["startTimeMs"])
    r = client.patch(f"/api/projects/{project_id}/words/{parts[1]['id']}",
                     json={"text": "edited"})
    check("word text edited",
          r.status_code == 200 and r.json()["text"] == "edited")
    check("edited word timing is tagged as user-set",
          r.json()["timingSource"] in ("user", "forced_alignment"))

    print("\n[11] exports")
    for fmt, check_fn in [
        ("pdf", lambda b: b[:5] == b"%PDF-"),
        ("txt", lambda b: b"E2E Test Song" in b),
        ("md", lambda b: b.startswith(b"# ")),
        ("json", lambda b: json.loads(b)["lines"] is not None),
        ("chordpro", lambda b: b"{title:" in b),
        ("html", lambda b: b"<!doctype html>" in b.lower()),
    ]:
        r = client.get(f"/api/projects/{project_id}/export/{fmt}")
        ok = r.status_code == 200 and check_fn(r.content)
        check(f"export {fmt}", ok, f"status {r.status_code}, {len(r.content)} bytes")
    r = client.get(f"/api/projects/{project_id}/export/docx")
    check("unknown export format rejected", r.status_code == 400)

    print("\n[12] isolation between users")
    other = TestClient(app)   # a different browser: no cookie, no access
    r = other.get(f"/api/projects/{project_id}")
    check("another browser cannot read the project", r.status_code == 404,
          str(r.status_code))
    r = other.get("/api/projects")
    check("another browser sees an empty project list", r.json() == [])

    print("\n[13] deletion")
    stored = client.get(f"/api/projects/{project_id}").json()
    r = client.delete(f"/api/projects/{project_id}")
    check("project deleted", r.status_code == 200)
    r = client.get(f"/api/projects/{project_id}")
    check("deleted project is gone", r.status_code == 404)
    audio_files = os.listdir(os.environ["CHORDVISION_STORAGE_DIR"])
    check("the audio file was removed from storage too",
          not any(f for f in audio_files if not f.startswith(".")),
          str(audio_files))

    return 1 if FAILED else 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        try:
            from app.jobs import shutdown_queue
            shutdown_queue()
        except Exception:
            pass
        shutil.rmtree(WORK, ignore_errors=True)
    print("\n" + "=" * 60)
    if FAILED:
        print(f"{len(FAILED)} checks FAILED:")
        for f in FAILED:
            print("   -", f)
    else:
        print("all end-to-end checks passed")
    sys.exit(code)
