"""A held-out test set, used only to check that tuning did not overfit.

The parameters in the engine were chosen by sweeping them against the set
in ``benchmark.py``.  A score measured on the same material that chose the
parameters is not evidence of anything - it is the definition of
overfitting.  This file contains different progressions, different keys,
different tempos, different instruments and different render settings,
and it is never used for tuning.  The gap between the two scores is the
honest estimate of how much of the benchmark number is real.
"""
from __future__ import annotations

import sys, os, time
from typing import List, Dict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "engine"))

import numpy as np
import synth
import chordvision_engine as cv
from chordvision_engine.theory import name_to_pc, transpose_chord
from evaluate import (
    Interval, chord_symbol_recall, boundary_scores, key_score,
    ItemResult, aggregate, VOCAB_LEVELS,
)
from benchmark import reference_intervals, estimate_intervals, print_summary

R = synth.RenderOptions

# Progressions that appear nowhere in benchmark.py.
HOLDOUT = [
    # (name, progression, tonic, mode, render options)
    ("andalusian",   ["Am", "G", "F", "E"],                "A", "minor",
     R(timbre="guitar", bpm=118, with_drums=True, strum_spread_s=0.022, seed=101)),
    ("canon",        ["D", "A", "Bm", "F#m", "G", "D", "G", "A"], "D", "major",
     R(timbre="strings", bpm=72, with_drums=False, with_reverb=True, reverb_wet=0.3, seed=102)),
    ("rock_IV_I_V",  ["F", "C", "G", "C"],                 "C", "major",
     R(timbre="guitar", bpm=140, with_drums=True, drum_level=0.5, seed=103)),
    ("bossa",        ["Fmaj7", "Bm7b5", "E7", "Am7"],      "A", "minor",
     R(timbre="epiano", bpm=128, walking_bass=True, with_drums=True, drum_level=0.25, seed=104)),
    ("gospel",       ["Eb", "Ab", "Eb", "Bb7"],            "D#", "major",
     R(timbre="organ", bpm=84, with_drums=True, with_reverb=True, reverb_wet=0.28, seed=105)),
    ("minor_blues",  ["Cm7", "Fm7", "Cm7", "G7"],          "C", "minor",
     R(timbre="epiano", bpm=96, walking_bass=True, with_drums=True, seed=106)),
    ("folk_capo",    ["G", "Em", "C", "D"],                "G", "major",
     R(timbre="guitar", bpm=108, strum_spread_s=0.025, with_drums=False,
       tuning_cents=-18.0, seed=107)),
    ("ballad_slow",  ["Bb", "Gm", "Eb", "F"],              "A#", "major",
     R(timbre="piano", bpm=58, with_drums=False, with_reverb=True,
       reverb_rt60=1.8, reverb_wet=0.33, seed=108)),
    ("funk_fast",    ["Em7", "A7", "Dmaj7", "Dmaj7"],      "D", "major",
     R(timbre="guitar", bpm=116, beats_per_chord=2, with_drums=True,
       drum_level=0.5, seed=109)),
    ("sus_pop",      ["Dsus4", "D", "Asus4", "A"],         "D", "major",
     R(timbre="piano", bpm=94, with_drums=True, seed=110)),
    ("chromatic_2",  ["F", "F#dim", "C/G", "G7"],          "C", "major",
     R(timbre="piano", bpm=88, with_drums=False, with_reverb=True, seed=111)),
    ("noisy_lofi",   ["Am", "C", "G", "Dm"],               "A", "minor",
     R(timbre="organ", bpm=102, with_drums=True, drum_level=0.6,
       noise_db=-28.0, tuning_cents=27.0, seed=112)),
]


def run(verbose: bool = True):
    engine = cv.ChordEngine()
    scratch = os.environ.get("CHORDVISION_SCRATCH", "/tmp")
    os.makedirs(scratch, exist_ok=True)
    results: List[ItemResult] = []

    if verbose:
        print(f"{'item':22s} {'root':>6s} {'majmin':>7s} {'triad':>7s} "
              f"{'7ths':>6s} {'bass':>6s} {'bndF1':>6s} {'key':>5s}  detected key")
        print("-" * 96)

    for name, prog, tonic, mode, opts in HOLDOUT:
        full = prog * 2
        y, sr, _ = synth.render_progression(full, opts)
        wav = os.path.join(scratch, f"holdout_{name}.wav")
        synth.write_wav(wav, y, sr)

        t0 = time.time()
        res = engine.analyze_file(wav)
        analysis_s = time.time() - t0

        ref = reference_intervals(full, opts)
        est = estimate_intervals(res)
        csr = {lv: chord_symbol_recall(ref, est, lv)[0] for lv in VOCAB_LEVELS}
        bnd = boundary_scores(ref, est, 0.25)
        kp = key_score(name_to_pc(tonic), mode, name_to_pc(res.key.tonic), res.key.mode)

        results.append(ItemResult(
            name=name, csr=csr, boundaries=bnd, key_points=kp,
            key_ref=f"{tonic} {mode}", key_est=res.key.name,
            duration_s=len(y) / sr, analysis_s=analysis_s,
            tuning_cents_ref=opts.tuning_cents,
            tuning_cents_est=res.tuning_offset_cents,
        ))

        if verbose:
            print(f"{name:22s} {csr['root']:6.3f} {csr['majmin']:7.3f} {csr['triad']:7.3f} "
                  f"{csr['sevenths']:6.3f} {csr['bass']:6.3f} {bnd['f1']:6.3f} {kp:5.2f}  "
                  f"{res.key.name} (truth {tonic} {mode})")
        try:
            os.remove(wav)
        except OSError:
            pass

    return results, aggregate(results)


if __name__ == "__main__":
    print("HELD-OUT TEST SET  (never used for tuning)\n")
    _r, summary = run()
    print_summary(summary)
