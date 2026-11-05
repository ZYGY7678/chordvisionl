"""Unit tests for the chord engine.

Run with:  python3 -m pytest tests/test_engine.py -v
       or:  python3 tests/test_engine.py       (no pytest needed)
"""
from __future__ import annotations

import sys, os, math, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "engine"))

import numpy as np

import synth
import chordvision_engine as cv
from chordvision_engine.cqt import cqt, harmonic_percussive_split
from chordvision_engine.chroma import estimate_tuning, chromagram, normalize_frames
from chordvision_engine.theory import (
    parse_chord, chord_name, transpose_chord, build_templates, pc_name,
    name_to_pc, CHORD_QUALITIES, QUALITY_SETS,
)
from chordvision_engine.decode import (
    DecoderConfig, decode_chords, path_to_segments, merge_short_segments,
)
from chordvision_engine.rhythm import onset_strength, estimate_tempo, snap_to_onsets
from chordvision_engine.harmony import estimate_key_from_chords
from chordvision_engine.audio import AudioDecodeError, probe, decode


SR = 11025


# --- CQT --------------------------------------------------------------------

def test_cqt_locates_pitch_exactly():
    """A pure tone must land in the bin for its own frequency."""
    t = np.arange(2 * SR) / SR
    for hz, label in [(65.406, "C2"), (261.626, "C4"), (440.0, "A4"), (1760.0, "A6")]:
        y = 0.7 * np.sin(2 * np.pi * hz * t)
        C, spec = cqt(y, SR)
        freqs = spec.bin_frequencies()
        peak = int(np.argmax(C[:, C.shape[1] // 2]))
        err_cents = 1200 * math.log2(freqs[peak] / hz)
        assert abs(err_cents) < 17, f"{label}: off by {err_cents:.1f} cents"


def test_cqt_amplitude_is_register_independent():
    """The same amplitude must read the same at the bottom and the top."""
    t = np.arange(2 * SR) / SR
    mags = []
    for hz in (82.41, 329.63, 1318.5):
        y = 0.5 * np.sin(2 * np.pi * hz * t)
        C, _ = cqt(y, SR)
        mags.append(float(C[:, C.shape[1] // 2].max()))
    assert max(mags) / min(mags) < 1.25, f"magnitudes vary too much: {mags}"


def test_cqt_rejects_bad_parameters():
    y = np.zeros(SR)
    for kwargs in (
        {"n_bins": 250},          # not a whole number of octaves
        {"hop_length": 500},      # not divisible down the octave chain
        {"n_bins": 36 * 9},       # highest bin above Nyquist
    ):
        try:
            cqt(y, SR, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"cqt should have rejected {kwargs}")


def test_hpss_separates_a_click_from_a_tone():
    t = np.arange(3 * SR) / SR
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    clicks = np.zeros_like(tone)
    for k in range(1, 6):
        at = int(k * 0.5 * SR)
        clicks[at:at + 200] = np.random.default_rng(0).normal(0, 0.8, 200)
    C, _ = cqt(tone + clicks, SR)
    H, P = harmonic_percussive_split(C)
    # The tone's bin should keep most of its energy in the harmonic part.
    bin440 = int(np.argmax(C.mean(axis=1)))
    assert H[bin440].sum() > 3 * P[bin440].sum()


# --- tuning -----------------------------------------------------------------

def test_tuning_estimator_recovers_known_offsets():
    for true_cents in (-45.0, -20.0, 0.0, 18.0, 44.0):
        y, sr, _ = synth.render_progression(
            ["C", "F"], synth.RenderOptions(
                sr=SR, bpm=120, with_drums=False, with_reverb=False,
                tuning_cents=true_cents, seed=3),
        )
        C, spec = cqt(y.astype(np.float64), sr)
        est = estimate_tuning(C, spec)
        assert abs(est - true_cents) < 8, \
            f"tuning: expected {true_cents}, got {est:.1f}"


# --- theory -----------------------------------------------------------------

def test_chord_symbol_round_trip():
    for symbol in ["C", "Am", "G7", "Fmaj7", "Bm7b5", "C#dim7", "Dsus4",
                   "Ebm9", "A13", "F#m6", "Bbadd9"]:
        root, quality, bass = parse_chord(symbol)
        rebuilt = chord_name(root, quality, bass)
        assert parse_chord(rebuilt) == (root, quality, bass), \
            f"{symbol} -> {rebuilt} did not round trip"


def test_transpose_is_cyclic():
    for symbol in ["C", "Am7", "G/B", "F#maj7"]:
        out = symbol
        for _ in range(12):
            out = transpose_chord(out, 1)
        assert parse_chord(out) == parse_chord(symbol), \
            f"{symbol} did not return after 12 semitones, got {out}"


def test_transpose_moves_the_bass_too():
    assert parse_chord(transpose_chord("G/B", 2)) == parse_chord("A/C#")


def test_templates_are_unit_vectors_and_distinct():
    T, labels, priors = build_templates(QUALITY_SETS["sevenths"])
    norms = np.sqrt((T * T).sum(axis=1))
    assert np.allclose(norms, 1.0, atol=1e-9)
    # No two chords may share a template, or they could never be told apart.
    G = T @ T.T
    np.fill_diagonal(G, 0.0)
    assert G.max() < 0.999, "two chord templates are identical"
    assert len(labels) == len(priors) == T.shape[0]


def test_major_and_major_seventh_are_separable():
    """The whole point of the tone-weight term in the templates."""
    T, labels, _ = build_templates(QUALITY_SETS["sevenths"])
    c = labels.index((0, "maj")); c7 = labels.index((0, "maj7"))
    assert float(T[c] @ T[c7]) < 0.93


# --- decoding ---------------------------------------------------------------

def _noisy_held_chord(sigma: float, n_frames: int = 200):
    """A single held C major chord buried in additive noise."""
    T, labels, _ = build_templates(QUALITY_SETS["triads"])
    c_index = labels.index((0, "maj"))
    rng = np.random.default_rng(5)
    chroma = np.tile(T[c_index][:, None], (1, n_frames))
    chroma += rng.normal(0, sigma, chroma.shape)
    chroma = normalize_frames(np.maximum(chroma, 0), "l2")
    return chroma, labels, c_index


def test_decoder_prefers_stability_over_flicker():
    """Noisy per-frame evidence must not produce a chord change per frame.

    sigma = 0.25 leaves the frames at about 0.84 cosine similarity to the
    true chord, which is a realistic figure for a busy mix.  The decoder
    alone - before any segment merging - has to hold one chord across all
    200 frames.
    """
    chroma, labels, c_index = _noisy_held_chord(0.25)
    res = decode_chords(chroma, frame_duration_s=0.046,
                        cfg=DecoderConfig(quality_set="triads"))
    segs = path_to_segments(res.path)
    assert len(segs) == 1, f"decoder produced {len(segs)} segments for one chord"
    assert segs[0][2] == c_index, "decoder settled on the wrong chord"


def test_pipeline_survives_noise_that_breaks_the_raw_decoder():
    """At sigma = 0.35 the raw decoder does flicker, and that is correct.

    The frames there sit at roughly 0.75 cosine similarity, which is the
    point at which the evidence genuinely stops supporting a confident
    chord.  What the full pipeline must do is recover: merging short
    segments has to collapse the flicker back to the one chord that is
    really there, rather than emitting a stream of fragments.
    """
    chroma, labels, c_index = _noisy_held_chord(0.35)
    res = decode_chords(chroma, frame_duration_s=0.046,
                        cfg=DecoderConfig(quality_set="triads"))
    raw = path_to_segments(res.path)
    merged = merge_short_segments(raw, min_frames=10, posterior=res.posterior)
    assert len(merged) == 1, f"pipeline left {len(merged)} segments (raw had {len(raw)})"
    assert merged[0][2] == c_index


def test_silence_becomes_no_chord():
    chroma = np.zeros((12, 100))
    energy = np.zeros(100)
    chroma[:, 40:60] = 0.5
    energy[40:60] = 1.0
    chroma = normalize_frames(chroma, "l2")
    res = decode_chords(chroma, 0.046, energy=energy)
    assert res.path[0] == res.no_chord_index
    assert res.path[-1] == res.no_chord_index


def test_merge_short_segments_removes_fragments():
    segs = [(0, 50, 1), (50, 53, 2), (53, 100, 1)]
    merged = merge_short_segments(segs, min_frames=10)
    assert merged == [(0, 100, 1)], merged


def test_path_to_segments_round_trip():
    path = np.array([0, 0, 1, 1, 1, 2, 0, 0])
    segs = path_to_segments(path)
    assert segs == [(0, 2, 0), (2, 5, 1), (5, 6, 2), (6, 8, 0)]


# --- rhythm -----------------------------------------------------------------

def test_tempo_estimation_on_a_known_click_track():
    for bpm in (80.0, 100.0, 128.0):
        n = int(20 * SR)
        y = np.zeros(n)
        period = int(60.0 / bpm * SR)
        rng = np.random.default_rng(1)
        for at in range(0, n - 400, period):
            y[at:at + 400] += rng.normal(0, 1, 400) * np.exp(-np.arange(400) / 60)
        C, spec = cqt(y, SR)
        env = onset_strength(C)
        est = estimate_tempo(env, spec.frame_rate_hz)
        ratio = est / bpm
        assert min(abs(ratio - r) for r in (0.5, 1.0, 2.0)) < 0.06, \
            f"tempo {bpm}: got {est:.1f}"


def test_snap_never_moves_a_boundary_too_far():
    boundaries = np.array([10, 50, 90])
    onsets = np.array([11, 70, 89])
    out = snap_to_onsets(boundaries, onsets, max_shift_frames=3)
    assert out[0] == 11          # moved, within range
    assert out[1] == 50          # left alone, nearest onset is 20 frames away
    assert out[2] == 89          # moved, within range


# --- key --------------------------------------------------------------------

def test_key_tells_relative_major_from_minor():
    """The case a pitch histogram alone can never solve."""
    a_minor = [(4.0, name_to_pc("A"), "min"), (4.0, name_to_pc("F"), "maj"),
               (4.0, name_to_pc("C"), "maj"), (4.0, name_to_pc("G"), "maj"),
               (4.0, name_to_pc("A"), "min")]
    k = estimate_key_from_chords(a_minor)
    assert (k.tonic, k.mode) == ("A", "minor"), f"got {k.name}"

    c_major = [(4.0, name_to_pc("C"), "maj"), (4.0, name_to_pc("G"), "maj"),
               (4.0, name_to_pc("A"), "min"), (4.0, name_to_pc("F"), "maj"),
               (4.0, name_to_pc("C"), "maj")]
    k = estimate_key_from_chords(c_major)
    assert (k.tonic, k.mode) == ("C", "major"), f"got {k.name}"


# --- audio io ---------------------------------------------------------------

def test_non_audio_file_is_rejected():
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"this is not audio, it only has an audio extension")
        path = f.name
    try:
        probe(path)
    except AudioDecodeError:
        return
    finally:
        os.unlink(path)
    raise AssertionError("a text file named .mp3 was accepted as audio")


def test_missing_file_is_rejected():
    try:
        probe("/nonexistent/path/to/nothing.wav")
    except AudioDecodeError:
        return
    raise AssertionError("a missing file was accepted")


# --- end to end -------------------------------------------------------------

def test_end_to_end_recovers_a_known_progression():
    progression = ["C", "G", "Am", "F"]
    y, sr, spans = synth.render_progression(
        progression * 2,
        synth.RenderOptions(bpm=100, with_drums=True, with_reverb=True, seed=9),
    )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        synth.write_wav(path, y, sr)
        res = cv.analyze_file(path)

        assert res.key.tonic == "C" and res.key.mode == "major", res.key.name
        assert res.tempo_bpm and abs(res.tempo_bpm - 100) < 8, res.tempo_bpm

        # Sample the detected chord at the middle of each reference span.
        for span in spans:
            mid_ms = (span.start_s + span.end_s) / 2 * 1000
            found = [c for c in res.chords
                     if c.start_time_ms <= mid_ms < c.end_time_ms]
            assert found, f"no chord covers {mid_ms:.0f} ms"
            assert found[0].root == pc_name(span.root_pc), \
                f"at {mid_ms:.0f} ms expected {span.symbol}, got {found[0].chord_name}"

        assert all(0.0 <= c.confidence <= 1.0 for c in res.chords)
        assert res.duration_ms > 0
        d = res.to_dict()
        assert "chords" in d and "key" in d
    finally:
        os.unlink(path)


def test_engine_never_reports_fabricated_certainty():
    """Confidence must be a real number in range, never a hardcoded 1.0."""
    y, sr, _ = synth.render_progression(
        ["C", "Am", "Dm7", "G7"],
        synth.RenderOptions(bpm=110, with_drums=True, noise_db=-26.0, seed=11),
    )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        synth.write_wav(path, y, sr)
        res = cv.analyze_file(path)
        confs = [c.confidence for c in res.chords]
        assert all(0.0 <= c <= 1.0 for c in confs)
        assert not all(c == 1.0 for c in confs), "confidence is not being computed"
        for c in res.chords:
            for alt in c.alternatives:
                assert alt.chord_name != c.chord_name
    finally:
        os.unlink(path)


# --- runner -----------------------------------------------------------------

if __name__ == "__main__":
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
