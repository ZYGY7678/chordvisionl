"""Parameter sweep over the engine configuration.

The test material is rendered once and reused, so only the analysis is
repeated.  Every configuration is scored on the *same* audio, which is
what makes the comparison meaningful.
"""
from __future__ import annotations

import sys, os, json, time, itertools, copy
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "engine"))

import numpy as np
import synth
import chordvision_engine as cv
from chordvision_engine.analyze import EngineConfig
from chordvision_engine.decode import DecoderConfig
from chordvision_engine.theory import name_to_pc
from benchmark import build_test_set, reference_intervals, estimate_intervals
from evaluate import chord_symbol_recall, boundary_scores, key_score, VOCAB_LEVELS

CACHE = os.environ.get("CHORDVISION_SCRATCH", "/tmp") + "/sweep_cache"


def prepare(quick: bool = False) -> List[dict]:
    os.makedirs(CACHE, exist_ok=True)
    items = build_test_set(quick)
    for it in items:
        wav = os.path.join(CACHE, f"{it['name']}.wav")
        if not os.path.exists(wav):
            y, sr, _ = synth.render_progression(it["progression"], it["options"])
            synth.write_wav(wav, y, sr)
        it["wav"] = wav
    return items


def score_config(items: List[dict], cfg: EngineConfig) -> Dict[str, float]:
    engine = cv.ChordEngine(cfg)
    rows = []
    t0 = time.time()
    for it in items:
        res = engine.analyze_file(it["wav"])
        ref = reference_intervals(it["progression"], it["options"])
        est = estimate_intervals(res)
        csr = {lv: chord_symbol_recall(ref, est, lv)[0] for lv in VOCAB_LEVELS}
        bnd = boundary_scores(ref, est, 0.25)
        kp = key_score(it["tonic_pc"], it["mode"], name_to_pc(res.key.tonic), res.key.mode)
        dur = it["options"].beats_per_chord * 60.0 / it["options"].bpm * len(it["progression"])
        rows.append((csr, bnd, kp, dur, it["name"]))

    durs = np.array([r[3] for r in rows]); w = durs / durs.sum()
    out = {}
    for lv in VOCAB_LEVELS:
        out[lv] = float(np.sum([r[0][lv] for r in rows] * w))
    out["bnd_f1"] = float(np.sum([r[1]["f1"] for r in rows] * w))
    out["bnd_prec"] = float(np.sum([r[1]["precision"] for r in rows] * w))
    out["key"] = float(np.mean([r[2] for r in rows]))
    out["worst7"] = float(min(r[0]["sevenths"] for r in rows))
    out["worst_name"] = min(rows, key=lambda r: r[0]["sevenths"])[4]
    out["seconds"] = round(time.time() - t0, 1)
    # A single figure to rank by: the strict vocabularies and timing are
    # what actually matter, so they carry the weight.
    out["objective"] = (0.30 * out["majmin"] + 0.34 * out["sevenths"]
                        + 0.16 * out["bass"] + 0.14 * out["bnd_f1"] + 0.06 * out["key"])
    return out


def make_cfg(**kw) -> EngineConfig:
    dec = DecoderConfig(
        quality_set=kw.get("quality_set", "sevenths"),
        kappa=kw.get("kappa", 30.0),
        mean_chord_duration_s=kw.get("mean_chord_duration_s", 2.6),
        fifths_tau=kw.get("fifths_tau", 1.9),
        same_root_weight=kw.get("same_root_weight", 0.75),
        no_chord_similarity=kw.get("no_chord_similarity", 0.74),
        min_segment_s=kw.get("min_segment_s", 0.45),
        bass_root_weight=kw.get("bass_root_weight", 0.0),
    )
    return EngineConfig(
        decoder=dec,
        chroma_smooth_frames=kw.get("chroma_smooth_frames", 13),
        use_hpss=kw.get("use_hpss", True),
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=str, default="core")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    items = prepare(args.quick)
    print(f"test set: {len(items)} items, cached in {CACHE}\n")

    if args.grid == "core":
        grid = {
            "mean_chord_duration_s": [1.8, 2.6, 3.4],
            "kappa": [18.0, 22.0, 28.0],
            "min_segment_s": [0.28, 0.5],
        }
    elif args.grid == "chroma":
        grid = {
            "chroma_smooth_frames": [5, 9, 13, 17],
            "use_hpss": [True, False],
        }
    else:
        grid = json.loads(args.grid)

    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"{len(combos)} configurations\n")
    header = "  ".join(f"{k[:14]:>14s}" for k in keys)
    print(f"{header}  {'majmin':>7s} {'7ths':>7s} {'bass':>7s} {'bndF1':>7s} {'key':>5s} {'OBJ':>7s}  worst")
    print("-" * (len(header) + 60))

    results = []
    for combo in combos:
        kw = dict(zip(keys, combo))
        sc = score_config(items, make_cfg(**kw))
        results.append((sc["objective"], kw, sc))
        vals = "  ".join(f"{str(v)[:14]:>14s}" for v in combo)
        print(f"{vals}  {sc['majmin']:7.4f} {sc['sevenths']:7.4f} {sc['bass']:7.4f} "
              f"{sc['bnd_f1']:7.4f} {sc['key']:5.2f} {sc['objective']:7.4f}  "
              f"{sc['worst_name'][:22]} {sc['worst7']:.2f}")

    results.sort(reverse=True, key=lambda r: r[0])
    print("\nBEST:", json.dumps(results[0][1]), f"objective={results[0][0]:.4f}")


if __name__ == "__main__":
    main()
