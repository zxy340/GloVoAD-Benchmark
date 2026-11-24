"""
Revision rerun, step 2d: two supplementary analyses of the input-level masking.

  1. Corpus identification from the speech-only and the pause-only input vectors,
     with the logistic regression, folds and seeds that evaluate_v2.py used for the
     full representation (results_v2/corpus_identification.csv).
  2. Whether the gap between speech-only and pause-only F1 differs between models:
     paired bootstrap of (speech - pause)_A - (speech - pause)_B for every pair of
     models, on the out-of-fold predictions of evaluate_inputmask.py, with the same
     participant resamples and Holm adjustment within metric.

Usage:
  python supp_inputmask.py --work work/ --res results_inmask
"""
import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from evaluate_v2 import CLASSES, SEEDS, N_FOLDS, MODELS, Scorer, holm, load, lr_cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--res", default="results_inmask")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    work = Path(args.work)
    out = work / args.res

    idx, meta, feats, _, _ = load(work)
    groups = np.array([r["participant"] for r in meta])
    corpus = np.array([r["corpus"] for r in meta])
    n = len(idx)

    # ------------------------------------------------------------------ 1. corpus identification
    cl = np.unique(corpus, return_inverse=True)[1]
    cfolds = np.full((len(SEEDS), n), -1)
    for s, seed in enumerate(SEEDS):
        for k, (_, te) in enumerate(StratifiedGroupKFold(N_FOLDS, shuffle=True,
                                                          random_state=seed).split(cl, cl, groups)):
            cfolds[s, te] = k
    nco = cl.max() + 1
    sets = ["full_valid", "in_speech", "in_pause", "in_speech_all", "in_pause_all"]
    X = {}
    for m in MODELS:
        z = np.load(work / "features_inmask" / f"{m}.main.npz")
        pos = {v: k for k, v in enumerate(z["idx"].tolist())}
        sel = np.array([pos[i] for i in idx])
        X[m] = {"full_valid": feats[m]["full_valid"], **{k: z[k][sel] for k in sets[1:]}}
    todo = [(m, k) for m in MODELS for k in sets]
    res = Parallel(n_jobs=args.jobs)(delayed(lr_cv)(X[m][k], cl, cfolds, nc=nco) for m, k in todo)
    rows = []
    for (m, k), P in zip(todo, res):
        rows.append({"model": m, "input": k,
                     "corpus_macro_f1": round(float(np.mean([f1_score(cl, p, average="macro") for p in P])), 4),
                     "corpus_acc": round(float(np.mean([(p == cl).mean() for p in P])), 4)})
    with open(out / "corpus_identification_inmask.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "input", "corpus_macro_f1", "corpus_acc"])
        w.writeheader()
        w.writerows(rows)
    ref = {r["model"]: float(r["corpus_acc"]) for r in csv.DictReader(open(work / "results_v2" / "corpus_identification.csv"))}
    print("corpus identification accuracy (full must equal the benchmark run)")
    for m in MODELS:
        d = {r["input"]: r["corpus_acc"] for r in rows if r["model"] == m}
        assert abs(d["full_valid"] - ref[m]) < 1e-9, (m, d["full_valid"], ref[m])
        print(f"  {m:15s} full {d['full_valid']:.3f}  speech-only {d['in_speech']:.3f}  pause-only {d['in_pause']:.3f}"
              f"   (all-frame pooling: {d['in_speech_all']:.3f} / {d['in_pause_all']:.3f})")

    # ------------------------------------------------------------------ 2. does the gap differ between models
    z = np.load(out / "oof_predictions.npz")
    assert np.array_equal(z["idx"], idx)
    sc = Scorer(z["y"], z["groups"], z["folds"], args.boot)
    names = CLASSES + ["macro"]
    prow = []
    for suf in ["", "_all"]:
        gap_pt, gap_b = {}, {}
        for m in MODELS:
            pa, ba = sc.boot(z[f"tt_in_speech{suf}|in_speech{suf}|{m}"])
            pb, bb = sc.boot(z[f"tt_in_pause{suf}|in_pause{suf}|{m}"])
            gap_pt[m], gap_b[m] = pa - pb, ba - bb
        for a, b in combinations(MODELS, 2):
            d = gap_b[a] - gap_b[b]
            lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
            p = np.minimum(1.0, np.maximum(2 * np.minimum((d <= 0).mean(0), (d >= 0).mean(0)), 1.0 / len(d)))
            for j, name in enumerate(names):
                prow.append({"pooling": "retained" if suf == "" else "all", "A": a, "B": b, "metric": name,
                             "gap_A": round(float(gap_pt[a][j]), 4), "gap_B": round(float(gap_pt[b][j]), 4),
                             "diff": round(float(gap_pt[a][j] - gap_pt[b][j]), 4),
                             "lo": round(float(lo[j]), 4), "hi": round(float(hi[j]), 4), "p": float(p[j])})
    for pool in ["retained", "all"]:
        for name in names:
            ids = [i for i, r in enumerate(prow) if r["pooling"] == pool and r["metric"] == name]
            for i, a in zip(ids, holm([prow[i]["p"] for i in ids])):
                prow[i]["p_holm"] = round(float(a), 4)
                prow[i]["p"] = round(prow[i]["p"], 4)
    with open(out / "gap_between_models.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pooling", "A", "B", "metric", "gap_A", "gap_B", "diff", "lo", "hi", "p", "p_holm"])
        w.writeheader()
        w.writerows(prow)
    for pool in ["retained", "all"]:
        print(f"\nspeech-minus-pause gap, differences between models ({pool}-frame pooling), Holm over 28 pairs")
        for name in ["macro", "MCI", "AD"]:
            sig = [r for r in prow if r["pooling"] == pool and r["metric"] == name and r["p_holm"] < 0.05]
            print(f"  {name}: {len(sig)} of 28 pairs differ")
            for r in sig:
                print(f"     {r['A']:15s} {r['gap_A']:+.3f}  vs  {r['B']:15s} {r['gap_B']:+.3f}   "
                      f"diff {r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] p_holm={r['p_holm']:.3f}")


if __name__ == "__main__":
    main()
