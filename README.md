# ChordVision

Upload a recording, get its chords and its lyrics, with the chord symbols
sitting over the words where the changes actually happen, and a player
that keeps the two locked to the audio.

The chord recognition is built into this project. There is no chord
recognition API anywhere in the stack: every chord in the output is
derived from spectral analysis of the file you uploaded. Lyric
transcription is the one thing that goes elsewhere, because it has to —
it runs through a provider of your choosing, with an API key you supply.

---

## What it does

- **Chord recognition** across 12 roots and 19 qualities, with a bass
  note for inversions, a key estimate, and a confidence figure and
  alternatives for every chord.
- **Lyric transcription** through a provider you choose, with per-word
  timings, and forced alignment when the provider only times phrases.
- **Chord-over-lyric layout** driven by timestamps, not by dividing
  chords evenly across a line. Handles a chord changing in the middle of
  a word, and Hebrew and English in the same song.
- **A player** with 0.5x–1.5x speed that preserves pitch, a waveform you
  can scrub, section looping, and a highlight that follows the audio
  clock.
- **An editor** for every chord and every word: rename, retime, add,
  delete, split.
- **Transposition** between −12 and +12 semitones, which renames the
  chords and leaves the recording untouched.
- **Export** to PDF, ChordPro, plain text, Markdown, JSON, and a
  print-ready page.

---

## Running it

You need Python 3.11+, Node 18+, and **ffmpeg** on PATH. ffmpeg is not
optional; it is how audio is decoded.

```bash
git clone <this repo> && cd chordvision
make install          # Python packages and npm packages
./run.sh              # builds the frontend, then serves everything on :8000
```

`run.sh` writes a `.env` with a generated secret on first run. Open
<http://127.0.0.1:8000>.

For development with hot reload, run the two halves separately:

```bash
python3 -m uvicorn app.main:app --app-dir backend --reload   # API on :8000
cd frontend && npm run dev                                   # app on :5173
```

### Deploying

A `Dockerfile` and a Render Blueprint (`render.yaml`) are included. Step by step, in Hebrew: [`DEPLOY_RENDER.md`](DEPLOY_RENDER.md).

### Configuration

Everything is environment variables; `.env.example` documents all of
them. The two that matter:

| Variable | Why |
|---|---|
| `CHORDVISION_SECRET_KEY` | Encrypts stored API keys and signs the identity cookie. Required in production. Changing it invalidates both. |
| `CHORDVISION_DATABASE_URL` | SQLite by default so it runs with nothing installed. Use PostgreSQL for anything real. |

---

## Measured accuracy

Numbers rather than adjectives. The harness renders audio whose chords
are known exactly, runs the engine over it, and scores the result with
the standard MIREX metrics, weighted by duration.

The test set is split in two. Twenty-four items were used to choose the
engine's parameters. Twelve more — different progressions, keys, tempos
and instruments — were never used for tuning, and exist only to show
whether the tuning generalised.

| | tuned on | held out |
|---|---|---|
| Root only | 97.2% | 97.2% |
| Root + major/minor | 97.2% | 97.2% |
| Root + triad type | 97.0% | 97.2% |
| Full quality (7ths etc.) | 94.7% | **95.5%** |
| Full quality + inversion | 92.9% | 93.0% |
| Chord boundaries, F1 at ±250 ms | 92.4% | 93.4% |
| Median boundary error | 14 ms | 16 ms |
| Key, exact | 95.8% | 91.7% |
| Tuning estimate, mean error | 1.2 cents | 1.3 cents |

Roughly 90–100× real time on two cores: a four minute song takes a few
seconds.

The held-out set scores slightly *higher* than the set the parameters
were chosen on, which is the evidence that the parameters generalise
rather than fitting noise.

**What these numbers do not mean.** The test audio is synthesised. It has
realistic instrument spectra, inversions and voicings, drums, reverb,
walking bass, detuning and noise, and it is rendered to samples that the
analyser reads with no shortcut — but it has no distorted guitars with
shifting spectra, no vocals gliding between pitches, and no mastering
compression. Commercial recordings are harder. Published systems
evaluated on real annotated music report roughly 80–85% at the
major/minor level, and that is the range to expect here too.

Reproduce any of it:

```bash
make bench      # the tuning set
make holdout    # the held-out set
python3 tests/sweep.py --grid core   # re-derive the parameters
```

`docs/ACCURACY.md` has the per-item results and the known failure modes.

---

## How the chord engine works

```
audio file
  ├─ ffmpeg decode ──────────────► mono PCM at 11.025 kHz
  ├─ Constant-Q Transform ───────► 252 bins, C1–C8, 3 per semitone
  ├─ harmonic/percussive split ──► drums removed from the harmony
  ├─ tuning estimation ──────────► offset from A=440, in cents
  ├─ chromagram ─────────────────► 12 pitch classes per 46 ms frame
  ├─ template matching ──────────► 12 roots × 19 qualities
  ├─ Viterbi decoding ───────────► one chord path for the whole song
  ├─ onset snapping ─────────────► boundaries to real note attacks
  └─ bass + key estimation ──────► inversions, and the key
```

Four decisions carry most of the quality, and each is explained where it
is implemented:

**A Constant-Q Transform, not an STFT** (`engine/chordvision_engine/cqt.py`).
An STFT has one frequency resolution in Hz, but a semitone is 3.9 Hz at
C2 and 250 Hz at C7, so a single window is always wrong somewhere. The
CQT uses a window per bin. It is computed octave by octave with the
signal halved in rate each time, which keeps it O(n log n).

**Templates that model real instruments** (`theory.py`). A piano playing
C major also radiates a G and an E from the harmonics of its own notes,
so a binary chord template penalises the correct answer. The templates
sum a decaying harmonic series per chord tone, plus a separate term for
being an actual chord member — without that second term the templates for
C and Cmaj7 become nearly identical and the engine can never hear the
difference.

**Decoding the sequence, not the frame** (`decode.py`). Picking the best
chord per 46 ms frame produces output that flickers several times a
second. The song is modelled as a hidden Markov chain where staying put
is likely, and changes to nearby roots on the circle of fifths are likelier
than changes to distant ones. Viterbi then finds the best path for the
whole song at once, so a few bad frames are outvoted instead of punching
a hole in the result.

**Key from the chords, not the notes** (`harmony.py`). C major and A minor
contain exactly the same seven pitch classes, so no pitch histogram can
separate them. The key finder scores each of the 24 keys on how the
detected chords *function* in it, weighted by duration, with extra weight
on the first and last chord and on any dominant-to-tonic cadence.
Replacing the histogram method with this moved key detection from 66.7%
to 95.8%.

### Known failure modes, measured

- **A walking bass** pulls chords toward their upper structure: D minor 7
  over a bass walking D–F–A–C is genuinely a lot like F major, and the
  engine sometimes says F. This is the worst item in the test set, at 65%.
- **C6 and Am7 are the same four notes** (C E G A). No amount of spectral
  analysis separates them; only context does.
- Using the bass register to reinforce the root was tried and **made every
  metric worse** at every weight tested. The mechanism is still in the
  code, defaulted off, with the measurements in the comment, so nobody
  spends an afternoon rediscovering it.

---

## Transcription and your API key

Chord detection runs entirely on your server. Transcription cannot: it
needs a speech model, and this project does not ship one.

- You choose the provider and enter your own key.
- The key is encrypted with Fernet (AES-128-CBC + HMAC-SHA256) using a key
  derived from `CHORDVISION_SECRET_KEY` with PBKDF2, before it reaches the
  database.
- Only the last four characters are stored in clear, so the settings
  screen can show you which key is saved.
- It is never returned by any endpoint, never written to a log, and never
  stored in the browser.
- Audio is only sent to a provider after you turn that on explicitly, per
  account. The checkbox says which provider it is going to.

What this does not protect against, stated plainly: anyone holding both
the database and `CHORDVISION_SECRET_KEY` can decrypt the keys. Real
protection needs a key management service, and `backend/app/crypto.py` is
deliberately small enough to swap for one.

**Gemini (free).** The `gemini` provider uses Google's
`gemini-3.5-transcribe` with word-level timestamps and works with a free
key from https://aistudio.google.com/apikey. It is tested against a
simulated Gemini API (upload, retry, parsing), not against the live
service. The free tier has a small daily request limit.

**Status of the other provider adapters.** OpenAI Whisper, Deepgram and
AssemblyAI are written against each service's documented API and request
word-level timestamps. They have **not** been run against the live
services, because that needs paid keys this build does not have. Treat
them as reviewed but unverified. The fourth provider, `file`, reads a
transcript from a local JSON file; it is fully exercised by the test
suite and is also the only option if the recording must not leave your
machine.

### When a provider only times phrases

Then forced alignment runs (`backend/app/alignment.py`): a dynamic
program places the word boundaries inside the phrase using the voice-band
energy envelope and a syllable-count estimate of how long each word
should take.

This is not phoneme-level alignment — it has no acoustic model of the
words. On the test phrase it cut mean onset error from 225 ms (splitting
the phrase evenly) to 62 ms. Every word it produces is tagged
`forced_alignment` rather than `provider_word`, the API returns that tag,
and the interface marks estimated timings differently. An estimate is
never presented as a measurement.

---

## Project layout

```
engine/chordvision_engine/   the chord engine, numpy and scipy only
  cqt.py          Constant-Q Transform
  chroma.py       tuning estimation, chromagram
  theory.py       chords, templates, transposition
  decode.py       HMM + Viterbi
  harmony.py      key estimation, bass detection
  rhythm.py       onsets, tempo, beat tracking
  analyze.py      the pipeline

backend/app/                 FastAPI
  jobs.py         job queue; analysis runs in its own process
  alignment.py    forced alignment
  sync.py         chord-to-word placement
  export.py       PDF/ChordPro/text/Markdown/JSON/HTML
  crypto.py       API key encryption
  transcription/  provider interface and adapters

frontend/src/                React, TypeScript, Tailwind, Framer Motion

tests/
  test_engine.py   22 unit tests
  test_api_e2e.py  58 end-to-end checks against the running app
  benchmark.py     the tuning set
  holdout.py       the held-out set
  sweep.py         parameter sweeps
  synth.py         the audio renderer and its ground truth
```

---

## Tests

```bash
make test          # engine unit tests, end-to-end API tests, frontend typecheck
```

The end-to-end suite drives the real application over HTTP: it uploads a
real audio file, watches the worker, reads the analysis back, streams the
audio with range requests, transcribes with the file provider, checks the
synchronised view, transposes, edits chords and words, exports in all six
formats, confirms one browser cannot read another's projects, and deletes
the project along with its audio.

Two of its checks exist specifically to stop this project drifting into
the thing it was built not to be: one asserts that no chord ever reports a
confidence of exactly 1.0 from detection, and one asserts that words from
forced alignment are tagged as estimates.

---

## Security

- Uploads are rejected unless ffprobe can decode them as audio. A text
  file named `.mp3` does not get in; there is a test for it.
- Size and duration limits are enforced while streaming to disk, so an
  oversized upload cannot fill the disk before being rejected.
- Audio is served only through an endpoint that checks ownership, under an
  unguessable storage key.
- Projects are scoped to a signed, HTTP-only identity cookie. A request
  for someone else's project returns 404, not 403, so ids cannot be probed
  for existence.
- Rate limits per client on requests and on uploads.
- No secret has a working default.

Scope limits worth knowing before deploying: identity is a cookie rather
than an account, so there is no recovery and no second device; the job
queue is in-process, so more than one API instance needs Celery or RQ; the
rate limiter is in-memory, so it belongs in Redis or the load balancer
behind more than one instance.

---

## Licence and content

You are responsible for having the right to upload, transcribe and share
whatever you process.


## Production chord recognizer (v2)

The production analysis worker now uses the replacement recognizer supplied in
`chord-engine (1).zip`, integrated as `engine/chordvision_engine/recognizer_v2.py`
and adapted to ChordVision's existing result schema in `engine_adapter.py`. The
older in-project spectral engine remains in the repository for its isolated
unit/evaluation helpers, but it is no longer called by the production job queue.

The adapter preserves the existing API contract for chord events, alternatives,
confidence, key, tempo, beat/onset times and diagnostics. Audio decoding remains
on the existing ffmpeg validation path, so the accepted formats, upload checks,
transcription providers, word alignment, player, editing and exports remain
unchanged. The new engine reports measurable analysis phases to the existing
job-progress stream.

The replacement adds librosa, Numba and SoundFile dependencies. It is materially
more CPU- and memory-intensive than the previous in-project engine; the supplied
engine author estimates around 500 MB for its analysis process. Render's Free
instance has a 512 MB memory ceiling, so long or complex tracks may exceed the
available memory once the web server is included. The integration keeps the
current service plan and five-minute Render audio cap unchanged; a successful
local test does not guarantee that every track will fit on Render Free.
