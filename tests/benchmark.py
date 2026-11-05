"""Build the test set, run the engine over it, and report measured accuracy.

Run with:  python3 tests/benchmark.py [--quick] [--json out.json]
"""
from __future__ import annotations

import sys, os, json, time, argparse
from typing import List, Tuple, Dict, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "engine"))

import synth
from evaluate import (
    Interval, chord_symbol_recall, boundary_scores, key_score,
    ItemResult, aggregate, VOCAB_LEVELS,
)
import chordvision_engine as cv
from chordvision_engine.theory import name_to_pc, transpose_chord, NO_CHORD


# --- the test set -----------------------------------------------------------
# Progressions are written in C / Am and transposed, so the same harmonic
# material is tested in many keys without hand-writing each one.

PROGRESSIONS: Dict[str, Tuple[List[str], str, str]] = {
    # name:            (progression in C or Am,                 tonic, mode)
    "pop_I_V_vi_IV":   (["C", "G", "Am", "F"],                   "C", "major"),
    "doo_wop":         (["C", "Am", "F", "G"],                   "C", "major"),
    "minor_loop":      (["Am", "F", "C", "G"],                   "A", "minor"),
    "jazz_ii_V_I":     (["Dm7", "G7", "Cmaj7", "Cmaj7"],         "C", "major"),
    "jazz_turnaround": (["Cmaj7", "Am7", "Dm7", "G7"],           "C", "major"),
    "blues":           (["C7", "F7", "C7", "G7"],                "C", "major"),
    "sus_colour":      (["Csus4", "C", "Gsus4", "G"],            "C", "major"),
    "descending_bass": (["C", "G/B", "Am", "C/G"],               "C", "major"),
    "modal":           (["Am", "G", "F", "G"],                   "A", "minor"),
    "sixths":          (["C6", "Am7", "Dm7", "G7"],              "C", "major"),
    "extended":        (["Cmaj7", "F", "Em7", "Am"],             "C", "major"),
    "diminished":      (["C", "C#dim", "Dm", "G7"],              "C", "major"),
}


def _transpose(prog: List[str], semitones: int) -> List[str]:
    return [transpose_chord(c, semitones) for c in prog]


def build_test_set(quick: bool = False) -> List[dict]:
    """Define every test item: what to render and what the truth is."""
    items: List[dict] = []

    def add(name, prog, tonic, mode, opts, semitones=0):
        prog_t = _transpose(prog, semitones) if semitones else list(prog)
        tonic_pc = (name_to_pc(tonic) + semitones) % 12
        items.append({
            "name": name,
            "progression": prog_t * (2 if not quick else 1),
            "tonic_pc": tonic_pc,
            "mode": mode,
            "options": opts,
        })

    R = synth.RenderOptions

    # 1-4: the same pop progression, four keys, four timbres, clean.
    for i, (semi, timbre) in enumerate([(0, "piano"), (5, "guitar"),
                                        (7, "epiano"), (3, "organ")]):
        add(f"pop_clean_{timbre}_+{semi}", *PROGRESSIONS["pop_I_V_vi_IV"],
            R(timbre=timbre, bpm=92 + 6 * i, with_drums=False, with_reverb=True,
              reverb_wet=0.15, seed=10 + i), semitones=semi)

    # 5-8: full band - drums, reverb, strummed guitar.
    for i, (semi, timbre) in enumerate([(0, "guitar"), (2, "piano"),
                                        (9, "guitar"), (4, "strings")]):
        add(f"band_{timbre}_+{semi}", *PROGRESSIONS["doo_wop"],
            R(timbre=timbre, bpm=104 + 4 * i, with_drums=True, drum_level=0.45,
              strum_spread_s=0.018, with_reverb=True, reverb_wet=0.26,
              seed=20 + i), semitones=semi)

    # 9-11: minor keys.
    for i, semi in enumerate([0, 3, 8]):
        add(f"minor_+{semi}", *PROGRESSIONS["minor_loop"],
            R(timbre="piano", bpm=86 + 5 * i, with_drums=True, drum_level=0.3,
              with_reverb=True, seed=30 + i), semitones=semi)

    # 12-15: seventh chords, where the quality really has to be heard.
    for i, (key_name, semi) in enumerate([("jazz_ii_V_I", 0), ("jazz_turnaround", 5),
                                          ("blues", 2), ("sixths", 7)]):
        add(f"sevenths_{key_name}_+{semi}", *PROGRESSIONS[key_name],
            R(timbre="epiano", bpm=120 + 6 * i, with_drums=False,
              with_reverb=True, reverb_wet=0.18, walking_bass=True,
              seed=40 + i), semitones=semi)

    # 16-17: detuned recordings - the tuning estimator has to catch these.
    add("detuned_minus35", *PROGRESSIONS["pop_I_V_vi_IV"],
        R(timbre="piano", bpm=98, tuning_cents=-35.0, with_drums=True, seed=50))
    add("detuned_plus42", *PROGRESSIONS["doo_wop"],
        R(timbre="guitar", bpm=110, tuning_cents=42.0, with_drums=True, seed=51))

    # 18: inversions / slash chords.
    add("inversions", *PROGRESSIONS["descending_bass"],
        R(timbre="piano", bpm=76, with_drums=False, with_reverb=True, seed=60))

    # 19: suspended chords.
    add("suspensions", *PROGRESSIONS["sus_colour"],
        R(timbre="guitar", bpm=100, with_drums=True, strum_spread_s=0.02, seed=61))

    # 20: fast harmonic rhythm - two beats per chord instead of four.
    add("fast_changes", *PROGRESSIONS["doo_wop"],
        R(timbre="piano", bpm=132, beats_per_chord=2, with_drums=True, seed=62))

    # 21: slow, heavily reverberant - the hardest case for boundaries.
    add("slow_wet", *PROGRESSIONS["modal"],
        R(timbre="strings", bpm=62, with_drums=False, with_reverb=True,
          reverb_rt60=2.0, reverb_wet=0.4, seed=63))

    # 22: a passing diminished chord.
    add("chromatic", *PROGRESSIONS["diminished"],
        R(timbre="piano", bpm=96, beats_per_chord=2, with_drums=True, seed=64))

    # 23: noisy, low-fidelity source.
    add("noisy", *PROGRESSIONS["pop_I_V_vi_IV"],
        R(timbre="guitar", bpm=108, with_drums=True, drum_level=0.55,
          noise_db=-32.0, with_reverb=True, seed=65))

    # 24: extended chords over a busy bass.
    add("extended_walking", *PROGRESSIONS["extended"],
        R(timbre="epiano", bpm=88, walking_bass=True, with_drums=True, seed=66))

    if quick:
        items = items[:8]
    return items


def reference_intervals(progression: List[str], opts: synth.RenderOptions) -> List[Interval]:
    chord_dur = opts.beats_per_chord * 60.0 / opts.bpm
    out = []
    for i, sym in enumerate(progression):
        out.append(Interval.from_symbol(sym, i * chord_dur, (i + 1) * chord_dur))
    return out


def estimate_intervals(result) -> List[Interval]:
    from chordvision_engine.theory import parse_chord
    out = []
    for c in result.chords:
        if c.quality == NO_CHORD:
            out.append(Interval(c.start_time_ms / 1000.0, c.end_time_ms / 1000.0,
                                -1, NO_CHORD, None))
            continue
        r, q, b = parse_chord(c.chord_name)
        out.append(Interval(c.start_time_ms / 1000.0, c.end_time_ms / 1000.0, r, q, b))
    return out


def run(quick: bool = False, verbose: bool = True,
        config: Optional[cv.ChordEngine] = None,
        tolerance_s: float = 0.25) -> Tuple[List[ItemResult], Dict[str, object]]:
    items = build_test_set(quick)
    engine = config or cv.ChordEngine()
    results: List[ItemResult] = []

    if verbose:
        print(f"Rendering and analysing {len(items)} test items...\n")
        print(f"{'item':30s} {'root':>6s} {'majmin':>7s} {'triad':>7s} "
              f"{'7ths':>6s} {'bass':>6s} {'bndF1':>6s} {'key':>5s}")
        print("-" * 84)

    scratch = os.environ.get("CHORDVISION_SCRATCH", "/tmp")
    os.makedirs(scratch, exist_ok=True)

    for item in items:
        opts = item["options"]
        y, sr, _spans = synth.render_progression(item["progression"], opts)
        wav = os.path.join(scratch, f"bench_{item['name']}.wav")
        synth.write_wav(wav, y, sr)

        t0 = time.time()
        res = engine.analyze_file(wav)
        analysis_s = time.time() - t0

        ref = reference_intervals(item["progression"], opts)
        est = estimate_intervals(res)

        csr = {}
        for level in VOCAB_LEVELS:
            value, _dur = chord_symbol_recall(ref, est, level)
            csr[level] = value

        bnd = boundary_scores(ref, est, tolerance_s=tolerance_s)
        kp = key_score(item["tonic_pc"], item["mode"],
                       name_to_pc(res.key.tonic), res.key.mode)

        ir = ItemResult(
            name=item["name"],
            csr=csr,
            boundaries=bnd,
            key_points=kp,
            key_ref=f"{item['tonic_pc']} {item['mode']}",
            key_est=res.key.name,
            duration_s=len(y) / sr,
            analysis_s=analysis_s,
            tuning_cents_ref=opts.tuning_cents,
            tuning_cents_est=res.tuning_offset_cents,
        )
        results.append(ir)

        if verbose:
            print(f"{item['name'][:30]:30s} {csr['root']:6.3f} {csr['majmin']:7.3f} "
                  f"{csr['triad']:7.3f} {csr['sevenths']:6.3f} {csr['bass']:6.3f} "
                  f"{bnd['f1']:6.3f} {kp:5.2f}")

        try:
            os.remove(wav)
        except OSError:
            pass

    summary = aggregate(results)
    return results, summary


def print_summary(summary: Dict[str, object], tolerance_s: float = 0.25) -> None:
    print("\n" + "=" * 70)
    print("MEASURED ACCURACY  (synthetic test set, see tests/synth.py for caveats)")
    print("=" * 70)
    print(f"items: {summary['n_items']}   total audio: "
          f"{summary['total_duration_s']:.0f}s   speed: {summary['realtime_factor']}x realtime")
    print("\nChord Symbol Recall, duration weighted:")
    labels = {
        "root": "root only                ",
        "majmin": "root + major/minor       ",
        "triad": "root + triad type        ",
        "sevenths": "full quality (7ths etc.) ",
        "bass": "full quality + inversion ",
    }
    for level in VOCAB_LEVELS:
        print(f"  {labels[level]} {summary[f'csr_{level}']*100:6.2f}%"
              f"   (worst item {summary[f'csr_{level}_worst']*100:5.1f}%)")

    print(f"\nChord boundary placement (tolerance +/-{tolerance_s*1000:.0f} ms):")
    print(f"  precision {summary['boundary_precision']*100:6.2f}%"
          f"   recall {summary['boundary_recall']*100:6.2f}%"
          f"   F1 {summary['boundary_f1']*100:6.2f}%")
    if summary.get("boundary_median_error_ms") is not None:
        print(f"  median timing error: {summary['boundary_median_error_ms']:.0f} ms")

    print(f"\nKey detection:")
    print(f"  exact {summary['key_exact']*100:6.2f}%   MIREX-weighted "
          f"{summary['key_score']*100:6.2f}%")

    print(f"\nTuning estimation:")
    print(f"  mean absolute error {summary['tuning_mean_abs_error_cents']:.2f} cents"
          f"   (worst {summary['tuning_max_abs_error_cents']:.2f})")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--tolerance", type=float, default=0.25)
    args = ap.parse_args()

    results, summary = run(quick=args.quick, tolerance_s=args.tolerance)
    print_summary(summary, args.tolerance)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({
                "summary": summary,
                "items": [
                    {"name": r.name, "csr": r.csr, "boundaries": r.boundaries,
                     "key_points": r.key_points, "key_ref": r.key_ref,
                     "key_est": r.key_est, "duration_s": r.duration_s}
                    for r in results
                ],
            }, f, indent=2)
        print(f"\nwrote {args.json}")
