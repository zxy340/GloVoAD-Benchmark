"""
Revision rerun, step 2b: statistics for the input-level region masking.

Uses the classifier, folds, seeds, bootstrap and tests of evaluate_v2.py unchanged
(imported from it), so every number is paired with the benchmark results.

Feature sets (extract_inputmask.py):
  in_speech, in_pause               pauses / speech silenced in the waveform, mean over
                                    the frames of the retained region   (PRIMARY)
  in_shift_speech, in_shift_pause   same with the circularly shifted gate (control)
  *_all                             same inputs, mean over all frames   (sensitivity)

Analyses:
  pooled      five-class task on all corpora
                tt_*    train and test on the same gated representation
                bench   train on the full representation, test on the gated ones
  within      the same matched-training comparison inside each corpus that has at
              least two classes with >= 8 participants (as in evaluate_v2.py)

Usage:
  python evaluate_inputmask.py --work work/ --jobs 8 --boot 2000
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

from evaluate_v2 import (CLASSES, C, SEEDS, N_FOLDS, MODELS, Scorer, fit_predict, holm,
                         load, make_folds)

BASE = ["in_speech", "in_pause", "in_shift_speech", "in_shift_pause"]
SETS = BASE + [k + "_all" for k in BASE]


def load_inmask(work, idx, models, dry, feats, phases=("main", "shift")):
    """Gated-input features aligned to idx. --dry substitutes frame-level features
    (plumbing test only, results are meaningless and written elsewhere)."""
    out = {}
    for m in models:
        if dry:
            F = feats[m]
            d = {"in_speech": F["speech"], "in_pause": F["pause"],
                 "in_shift_speech": F["shift_speech"][:, 0], "in_shift_pause": F["shift_pause"][:, 0]}
            d.update({k + "_all": v for k, v in list(d.items())})
            out[m] = d
            continue
        d = {}
        for ph in phases:
            z = np.load(work / "features_inmask" / f"{m}.{ph}.npz")
            pos = {v: k for k, v in enumerate(z["idx"].tolist())}
            missing = [i for i in idx if i not in pos]
            if missing:
                raise SystemExit(f"{m}/{ph}: {len(missing)} analysed recordings have no gated features")
            sel = np.array([pos[i] for i in idx])
            for k in z.files:
                if k.startswith("in_"):
                    d[k] = z[k][sel]
        out[m] = d
    return out


def run_jobs(jobs, n_jobs):
    """jobs: list of (key, Xtr, ytr, [Xte...], nc, seed) -> {key: [pred arrays]}"""
    res = Parallel(n_jobs=n_jobs, verbose=1, pre_dispatch="2*n_jobs")(
        delayed(fit_predict)(Xtr, ytr, tests, nc, seed) for _, Xtr, ytr, tests, nc, seed in jobs)
    return {j[0]: r for j, r in zip(jobs, res)}


def add_summary(rows, sc, names, tag, exp, cond, model, P, boots, extra=None):
    pt, b = sc.boot(P)
    lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
    boots[(tag, exp, cond, model)] = (pt, b)
    for j, name in enumerate(names):
        r = {"experiment": exp, "condition": cond, "model": model, "metric": name,
             "point": round(float(pt[j]), 4), "lo": round(float(lo[j]), 4), "hi": round(float(hi[j]), 4)}
        if extra:
            r = {**extra, **r}
        rows.append(r)


def add_pair(pr, boots, names, tag, family, kA, kB, extra=None):
    (pA, bA), (pB, bB) = boots[(tag,) + kA], boots[(tag,) + kB]
    d, lo, hi, p = Scorer.paired(bA, bB, pA, pB)
    for j, name in enumerate(names):
        r = {"family": family, "A": "|".join(kA), "B": "|".join(kB), "metric": name,
             "diff": round(float(d[j]), 4), "lo": round(float(lo[j]), 4),
             "hi": round(float(hi[j]), 4), "p": float(p[j])}
        if extra:
            r = {**extra, **r}
        pr.append(r)


def finish_pairs(pr, key):
    fam = defaultdict(list)
    for i, r in enumerate(pr):
        fam[key(r)].append(i)
    for ids in fam.values():
        for i, a in zip(ids, holm([pr[i]["p"] for i in ids])):
            pr[i]["p_holm"] = round(float(a), 4)
            pr[i]["p"] = round(pr[i]["p"], 4)


def pair_plan(m, suf):
    """(family, A, B) for one model; suf is '' (primary) or '_all' (sensitivity)."""
    s, p = "in_speech" + suf, "in_pause" + suf
    ss, sp = "in_shift_speech" + suf, "in_shift_pause" + suf
    t = lambda k: ("tt_" + k, k, m)
    b = lambda k: ("bench", k, m)
    full = ("bench", "full_valid", m)
    return [("inmask_traintest" + suf, t(s), t(p)),
            ("inmask_traintest_vs_shift" + suf, t(s), t(ss)),
            ("inmask_traintest_vs_shift" + suf, t(p), t(sp)),
            ("inmask_traintest_vs_full" + suf, full, t(s)),
            ("inmask_traintest_vs_full" + suf, full, t(p)),
            ("inmask_trainfull" + suf, b(s), b(p)),
            ("inmask_trainfull_vs_shift" + suf, b(s), b(ss)),
            ("inmask_trainfull_vs_shift" + suf, b(p), b(sp))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--min_within", type=int, default=8)
    ap.add_argument("--models", default="all")
    ap.add_argument("--dry", default="", help="output dir for a plumbing test with stand-in features")
    ap.add_argument("--phases", default="main,shift",
                    help="'main' evaluates the speech-only and pause-only inputs before the shifted "
                         "control has been extracted (same analysis, written to results_inmask_main)")
    args = ap.parse_args()
    work = Path(args.work)
    phases = args.phases.split(",")
    assert "main" in phases and set(phases) <= {"main", "shift"}
    out = Path(args.dry) if args.dry else work / ("results_inmask" if "shift" in phases else "results_inmask_main")
    out.mkdir(parents=True, exist_ok=True)
    models = MODELS if args.models == "all" else args.models.split(",")
    base = BASE if "shift" in phases else BASE[:2]
    sets = base + [k + "_all" for k in base]            # SETS, or its speech/pause subset
    avail = set(sets) | {"full_valid"}

    idx, meta, feats, _, _ = load(work)
    G = load_inmask(work, idx, models, bool(args.dry), feats, phases)
    y = np.array([CLASSES.index(r["diagnosis"]) for r in meta])
    groups = np.array([r["participant"] for r in meta])
    corpus = np.array([r["corpus"] for r in meta])
    n = len(y)
    folds = make_folds(y, groups)
    sc = Scorer(y, groups, folds, args.boot)
    names = CLASSES + ["macro"]

    # ------------------------------------------------------------------ pooled task
    jobs = []
    for m in models:
        tests = {"full_valid": feats[m]["full_valid"], **{k: G[m][k] for k in sets}}
        for s in range(len(SEEDS)):
            for k in range(N_FOLDS):
                tr, te = folds[s] != k, folds[s] == k
                seed = SEEDS[s] * 100 + k
                jobs.append(((m, "bench", s, k), feats[m]["full_valid"][tr], y[tr],
                             [X[te] for X in tests.values()], C, seed))
                for key in sets:
                    X = G[m][key]
                    jobs.append(((m, "tt_" + key, s, k), X[tr], y[tr], [X[te]], C, seed))
    print(f"pooled task: {len(jobs)} MLP fits", flush=True)
    res = run_jobs(jobs, args.jobs)
    preds = {}
    for m in models:
        conds = ["full_valid"] + sets
        for cnd in conds:
            preds[("bench", cnd, m)] = np.full((len(SEEDS), n), -1)
        for key in sets:
            preds[("tt_" + key, key, m)] = np.full((len(SEEDS), n), -1)
        for s in range(len(SEEDS)):
            for k in range(N_FOLDS):
                te = folds[s] == k
                for cnd, p in zip(conds, res[(m, "bench", s, k)]):
                    preds[("bench", cnd, m)][s, te] = p
                for key in sets:
                    preds[("tt_" + key, key, m)][s, te] = res[(m, "tt_" + key, s, k)][0]

    # the full-representation classifier must reproduce the benchmark run exactly
    ref_path = work / "results_v2" / "oof_predictions.npz"
    if ref_path.exists():
        ref = np.load(ref_path)
        assert np.array_equal(ref["idx"], idx) and np.array_equal(ref["folds"], folds)
        for m in models:
            same = np.array_equal(ref[f"bench|full_valid|{m}"], preds[("bench", "full_valid", m)])
            print(f"  {m}: full-representation predictions identical to the benchmark run: {same}")
            assert same

    rows, boots, pr = [], {}, []
    for (exp, cnd, m), P in preds.items():
        add_summary(rows, sc, names, "pooled", exp, cnd, m, P, boots)
    for m in models:
        for suf in ["", "_all"]:
            for fam, kA, kB in pair_plan(m, suf):
                if kA[1] in avail and kB[1] in avail:
                    add_pair(pr, boots, names, "pooled", fam, kA, kB)
    finish_pairs(pr, lambda r: (r["family"], r["metric"]))
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "condition", "model", "metric", "point", "lo", "hi"])
        w.writeheader()
        w.writerows(rows)
    with open(out / "paired.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "A", "B", "metric", "diff", "lo", "hi", "p", "p_holm"])
        w.writeheader()
        w.writerows(pr)
    np.savez_compressed(out / "oof_predictions.npz", idx=idx, y=y, groups=groups, folds=folds,
                        **{"|".join(k): v for k, v in preds.items()})
    print("pooled task written", flush=True)

    # ------------------------------------------------------------------ within corpus
    wrows, wpr, info = [], [], {}
    wsets = ["full_valid"] + sets
    for c in sorted(set(corpus)):
        sel = corpus == c
        present = [i for i in range(C) if len(set(groups[sel & (y == i)])) >= args.min_within]
        if len(present) < 2:
            continue
        sel = sel & np.isin(y, present)
        yc = np.searchsorted(present, y[sel])
        gc = groups[sel]
        k_splits = min(N_FOLDS, min(len(set(gc[yc == j])) for j in range(len(present))))
        fl = np.full((len(SEEDS), sel.sum()), -1)
        for s, seed in enumerate(SEEDS):
            for k, (_, te) in enumerate(StratifiedGroupKFold(k_splits, shuffle=True,
                                                              random_state=seed).split(yc, yc, gc)):
                fl[s, te] = k
        scw = Scorer(yc, gc, fl, args.boot, nc=len(present))
        cnames = [CLASSES[i] for i in present] + ["macro"]
        extra = {"corpus": c, "classes": "/".join(CLASSES[i] for i in present), "n": int(sel.sum())}
        info[c] = {**extra, "participants": int(len(set(gc))), "k_splits": int(k_splits),
                   "class_recordings": {CLASSES[i]: int((yc == j).sum()) for j, i in enumerate(present)}}
        jobs = []
        for m in models:
            for key in wsets:
                X = (feats[m]["full_valid"] if key == "full_valid" else G[m][key])[sel]
                for s, seed in enumerate(SEEDS):
                    for k in range(k_splits):
                        jobs.append(((m, key, s, k), X[fl[s] != k], yc[fl[s] != k], [X[fl[s] == k]],
                                     len(present), seed * 100 + k))
        print(f"within {c}: {len(jobs)} MLP fits", flush=True)
        res = run_jobs(jobs, args.jobs)
        wb = {}
        for m in models:
            for key in wsets:
                P = np.full((len(SEEDS), sel.sum()), -1)
                for s in range(len(SEEDS)):
                    for k in range(k_splits):
                        P[s, fl[s] == k] = res[(m, key, s, k)][0]
                exp = "bench" if key == "full_valid" else "tt_" + key
                add_summary(wrows, scw, cnames, c, exp, key, m, P, wb, extra)
        for m in models:
            for suf in ["", "_all"]:
                for fam, kA, kB in pair_plan(m, suf):
                    if "trainfull" in fam:
                        continue  # only the matched-training design is run within corpora
                    if kA[1] in avail and kB[1] in avail:
                        add_pair(wpr, wb, cnames, c, fam, kA, kB, {"corpus": c})
    finish_pairs(wpr, lambda r: (r["corpus"], r["family"], r["metric"]))
    with open(out / "within_corpus.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["corpus", "classes", "n", "experiment", "condition", "model",
                                          "metric", "point", "lo", "hi"])
        w.writeheader()
        w.writerows(wrows)
    with open(out / "within_paired.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["corpus", "family", "A", "B", "metric", "diff", "lo", "hi", "p", "p_holm"])
        w.writeheader()
        w.writerows(wpr)
    (out / "run_info.json").write_text(json.dumps({"analysed": int(n), "models": models, "phases": phases,
                                                   "boot": args.boot, "within": info}, indent=1))
    print(f"Done. Results in {out}")


if __name__ == "__main__":
    main()
