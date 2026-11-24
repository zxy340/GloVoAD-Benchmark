"""
Revision rerun, step 2: every experiment and statistic for the revised manuscript.

Classifier, folds and seeds as in the paper (MLP 2x256, dropout 0.3, Adam 1e-3,
weight decay 1e-4, batch 64, 80 epochs, inverse-frequency class weights;
5-fold StratifiedGroupKFold x seeds 42/123/456), with these changes:
  * groups are participants for every corpus (Pitt ID no longer includes the label)
  * torch and data-loader seeds are fixed per fold, so reruns are reproducible
  * the same folds are reused for every model and condition (paired comparisons)
  * all out-of-fold predictions are saved

Point estimate: F1 on the pooled out-of-fold predictions of each seed, averaged
over seeds. Uncertainty: SD over the 15 fold-level F1 values, and a 95% CI from a
participant-level bootstrap of the pooled predictions (participants resampled
with replacement, same resample for all seeds and for both arms of a paired
comparison). Folds and seeds are never treated as independent samples.

Experiments (one MLP fit serves several test conditions):
  bench        train full_valid  -> test full_valid, speech, pause, shifted masks,
                                    reversed, locally reversed
  bench_all    train full_all    -> test full_all (Whisper pooling of the paper)
  tt_*         train and test on the same manipulated representation
               (speech, pause, shifted speech/pause, reversed, locally reversed)
  baselines    corpus label only, duration/VAD only, handcrafted librosa + LR,
               stratified dummy, analytic chance levels
  corpus_id    how well each embedding identifies the source corpus
  within       within-corpus classification for corpora with >= 2 eligible classes

Usage:
  python evaluate_v2.py --work work/ --jobs 8 --boot 2000
"""
import argparse
import csv
import itertools
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

CLASSES = ["Control", "MCI", "AD", "ProbableAD", "PPA"]
C = len(CLASSES)
SEEDS = [42, 123, 456]
N_FOLDS = 5
MODELS = ["wav2vec2_ssl", "wav2vec2", "hubert", "wavlm", "whisper_base", "xlsr", "whisper_medium", "unispeech_sat"]
WHISPER = {"whisper_base", "whisper_medium"}
SR = 16000


# ----------------------------------------------------------------------------- classifier
class MLP(nn.Module):
    def __init__(self, d, nc, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(h, nc))

    def forward(self, x):
        return self.net(x)


def fit_predict(Xtr, ytr, tests, nc, seed):
    """Fit scaler + MLP on (Xtr, ytr); return predictions for each test matrix."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    sc = StandardScaler().fit(Xtr)
    xt = torch.from_numpy(sc.transform(Xtr).astype(np.float32))
    yt = torch.from_numpy(ytr.astype(np.int64))
    cc = np.bincount(ytr, minlength=nc).astype(float)
    cw = 1.0 / (cc + 1e-6)
    cw = cw / cw.sum() * nc
    mdl = MLP(Xtr.shape[1], nc)
    opt = torch.optim.Adam(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32))
    loader = DataLoader(TensorDataset(xt, yt), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    mdl.train()
    for _ in range(80):
        for xb, yb in loader:
            loss = crit(mdl(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    mdl.eval()
    out = []
    with torch.no_grad():
        for X in tests:
            out.append(mdl(torch.from_numpy(sc.transform(X).astype(np.float32))).argmax(1).numpy())
    return out


# ----------------------------------------------------------------------------- statistics
class Scorer:
    """Seed-averaged pooled-OOF F1 with participant bootstrap, shared resamples."""

    def __init__(self, y, groups, folds, n_boot, nc=C, seed=2026):
        self.y, self.nc, self.folds = y, nc, folds  # folds: [S, N] fold id per seed
        _, self.g = np.unique(groups, return_inverse=True)
        G = self.g.max() + 1
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, G, size=(n_boot, G))
        self.W = np.stack([np.bincount(d, minlength=G) for d in draws]).astype(np.float32)
        self.Gm = np.zeros((G, len(y)), np.float32)
        self.Gm[self.g, np.arange(len(y))] = 1.0
        self.Y1 = np.eye(nc, dtype=np.float32)[y]

    def counts(self, P):  # P: [S, N] -> TP, FP, FN per seed, group, class
        out = []
        for p in P:
            P1 = np.eye(self.nc, dtype=np.float32)[p]
            tp = self.Gm @ (self.Y1 * P1)
            fp = self.Gm @ ((1 - self.Y1) * P1)
            fn = self.Gm @ (self.Y1 * (1 - P1))
            out.append((tp, fp, fn))
        return out

    @staticmethod
    def _f1(tp, fp, fn):
        den = 2 * tp + fp + fn
        return np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)

    def boot(self, P):
        """Return (point [nc+1], bootstrap [B, nc+1]); last column = macro-F1."""
        pts, bs = [], []
        for tp, fp, fn in self.counts(P):
            pts.append(self._f1(tp.sum(0), fp.sum(0), fn.sum(0)))
            bs.append(self._f1(self.W @ tp, self.W @ fp, self.W @ fn))
        pt, b = np.mean(pts, 0), np.mean(bs, 0)
        return np.append(pt, pt.mean()), np.concatenate([b, b.mean(1, keepdims=True)], 1)

    def fold_stats(self, P):
        vals = []
        for s, p in enumerate(P):
            for k in range(N_FOLDS):
                te = self.folds[s] == k
                f = f1_score(self.y[te], p[te], labels=range(self.nc), average=None, zero_division=0)
                vals.append(np.append(f, f.mean()))
        v = np.array(vals)
        return v.mean(0), v.std(0)

    def summary(self, P):
        pt, b = self.boot(P)
        fm, fs = self.fold_stats(P)
        lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
        return {"point": pt, "lo": lo, "hi": hi, "fold_mean": fm, "fold_sd": fs}, b

    @staticmethod
    def paired(bA, bB, ptA, ptB):
        d = bA - bB
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        p = 2 * np.minimum((d <= 0).mean(0), (d >= 0).mean(0))
        p = np.maximum(p, 1.0 / len(d))
        return ptA - ptB, lo, hi, np.minimum(p, 1.0)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


# ----------------------------------------------------------------------------- data
def load(work):
    with open(work / "manifest_v2.csv") as f:
        rows = {int(r["idx"]): r for r in csv.DictReader(f)}
    feats, common = {}, None
    for m in MODELS:
        z = np.load(work / "features_v2" / f"{m}.npz")
        feats[m] = z
        s = set(z["idx"].tolist())
        common = s if common is None else common & s
    idx = np.array(sorted(common))
    for m in MODELS:
        pos = {v: k for k, v in enumerate(feats[m]["idx"].tolist())}
        sel = np.array([pos[i] for i in idx])
        feats[m] = {k: feats[m][k][sel] for k in feats[m].files if k != "idx"}
    meta = [rows[i] for i in idx]
    n_ok = sum(r["status"] == "ok" for r in rows.values())
    return idx, meta, feats, len(rows), n_ok


def make_folds(y, groups):
    folds = np.full((len(SEEDS), len(y)), -1)
    for s, seed in enumerate(SEEDS):
        for k, (_, te) in enumerate(StratifiedGroupKFold(N_FOLDS, shuffle=True,
                                                          random_state=seed).split(y, y, groups)):
            folds[s, te] = k
    assert (folds >= 0).all()
    return folds


# ----------------------------------------------------------------------------- jobs
def neural_plan(feats):
    """(model, experiment, train matrix, {test name: test matrix}) for every model."""
    plan = []
    for m in MODELS:
        F = feats[m]
        tests = {"full_valid": F["full_valid"], "speech": F["speech"], "pause": F["pause"],
                 "rev_valid": F["rev_valid"], "lrev_valid": F["lrev_valid"]}
        for k in range(F["shift_speech"].shape[1]):
            tests[f"shift_speech_{k}"] = F["shift_speech"][:, k]
            tests[f"shift_pause_{k}"] = F["shift_pause"][:, k]
        plan.append((m, "bench", F["full_valid"], tests))
        if m in WHISPER:
            plan.append((m, "bench_all", F["full_all"], {"full_all": F["full_all"]}))
        for key in ["speech", "pause", "rev_valid", "lrev_valid"]:
            plan.append((m, f"tt_{key}", F[key], {key: F[key]}))
        plan.append((m, "tt_shift_speech", F["shift_speech"][:, 0], {"shift_speech_0": F["shift_speech"][:, 0]}))
        plan.append((m, "tt_shift_pause", F["shift_pause"][:, 0], {"shift_pause_0": F["shift_pause"][:, 0]}))
    return plan


def run_neural(plan, y, folds, n_jobs):
    """Fit every (plan entry, seed, fold); jobs are generated lazily to bound memory."""
    keys = [(pi, s, k) for pi in range(len(plan)) for s in range(len(SEEDS)) for k in range(N_FOLDS)]
    print(f"{len(keys)} MLP fits")

    def gen():
        for pi, s, k in keys:
            _, _, Xtr_all, tests = plan[pi]
            tr, te = folds[s] != k, folds[s] == k
            yield delayed(fit_predict)(Xtr_all[tr], y[tr], [X[te] for X in tests.values()],
                                       C, SEEDS[s] * 100 + k)

    res = Parallel(n_jobs=n_jobs, verbose=2, pre_dispatch="2*n_jobs")(gen())
    preds = {}
    for m, exp, _, tests in plan:
        for t in tests:
            preds[(exp, t, m)] = np.full((len(SEEDS), len(y)), -1)
    for (pi, s, k), outs in zip(keys, res):
        m, exp, _, tests = plan[pi]
        te = folds[s] == k
        for t, p in zip(tests, outs):
            preds[(exp, t, m)][s, te] = p
    return preds


def lr_cv(X, y, folds, nc=C, onehot=False):
    P = np.full((len(SEEDS), len(y)), -1)
    for s in range(len(SEEDS)):
        for k in range(N_FOLDS):
            tr, te = folds[s] != k, folds[s] == k
            if onehot:
                enc = OneHotEncoder(handle_unknown="ignore").fit(X[tr])
                Xtr, Xte = enc.transform(X[tr]), enc.transform(X[te])
            else:
                sc = StandardScaler().fit(X[tr])
                Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            lr = LogisticRegression(max_iter=3000, class_weight="balanced")
            P[s, te] = lr.fit(Xtr, y[tr]).predict(Xte)
    return P


def handcrafted_one(audio_path, n_rows, i, n):
    """The 37 librosa features of run_benchmark.py in the public repo, for one recording."""
    import librosa
    audio = np.memmap(audio_path, dtype="float32", mode="r", shape=(n_rows, SR * 30))
    y = np.array(audio[i, :n])
    mf = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13)
    f = list(mf.mean(1)) + list(librosa.feature.delta(mf).mean(1))
    f += [librosa.feature.spectral_centroid(y=y, sr=SR).mean(),
          librosa.feature.spectral_bandwidth(y=y, sr=SR).mean(),
          librosa.feature.spectral_contrast(y=y, sr=SR).mean(),
          librosa.feature.spectral_rolloff(y=y, sr=SR).mean(),
          librosa.feature.spectral_flatness(y=y).mean(),
          librosa.feature.zero_crossing_rate(y).mean(), librosa.feature.rms(y=y).mean(),
          librosa.feature.chroma_stft(y=y, sr=SR).mean()]
    tempo, _ = librosa.beat.beat_track(y=y, sr=SR)
    f.append(float(np.atleast_1d(tempo)[0]))
    pit, mag = librosa.piptrack(y=y, sr=SR)
    pv = pit[mag > np.median(mag)]
    pv = pv[pv > 0]
    f += [pv.mean() if len(pv) else 0.0, pv.std() if len(pv) else 0.0]
    return np.nan_to_num(np.array(f, np.float32))


def handcrafted(work, idx, meta, n_jobs):
    path = work / "features_v2" / "handcrafted.npz"
    if not path.exists():
        n_rows = sum(1 for _ in open(work / "manifest_v2.csv")) - 1
        X = np.stack(Parallel(n_jobs=n_jobs)(
            delayed(handcrafted_one)(str(work / "audio_30s.f32"), n_rows, int(r["idx"]),
                                     int(r["used_samples"])) for r in meta))
        np.savez(path, idx=idx, X=X)
    z = np.load(path)
    pos = {v: k for k, v in enumerate(z["idx"].tolist())}
    return z["X"][[pos[i] for i in idx]]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--min_within", type=int, default=8,
                    help="min participants per class for the within-corpus analysis")
    args = ap.parse_args()
    work = Path(args.work)
    out = work / "results_v2"
    out.mkdir(exist_ok=True)

    idx, meta, feats, n_files, n_ok = load(work)
    y = np.array([CLASSES.index(r["diagnosis"]) for r in meta])
    groups = np.array([r["participant"] for r in meta])
    corpus = np.array([r["corpus"] for r in meta])
    n = len(y)
    folds = make_folds(y, groups)
    sc = Scorer(y, groups, folds, args.boot)
    info = {"files_found": n_files, "usable_after_audio_checks": n_ok, "analysed": n,
            "participants": int(len(set(groups))),
            "class_recordings": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
            "class_participants": {c: int(len(set(groups[y == i]))) for i, c in enumerate(CLASSES)}}
    print(json.dumps(info, indent=1))

    summ_rows, boots = [], {}

    def add(exp, cond, model, P):
        s, b = sc.summary(P)
        boots[(exp, cond, model)] = (s["point"], b)
        for j, name in enumerate(CLASSES + ["macro"]):
            summ_rows.append({"experiment": exp, "condition": cond, "model": model, "metric": name,
                              **{k: round(float(v[j]), 4) for k, v in s.items()}})

    # neural experiments
    plan = neural_plan(feats)
    preds = run_neural(plan, y, folds, args.jobs)
    np.savez_compressed(out / "oof_predictions.npz", idx=idx, y=y, groups=groups, folds=folds,
                        **{"|".join(k): v for k, v in preds.items()})
    for (exp, cond, m), P in preds.items():
        if cond.startswith("shift_") and not cond.endswith("_0") and exp == "bench":
            continue  # summarised below as the mean over shifts
        add(exp, cond, m, P)
    # shifted masks under the main model: average F1 over the K shifts
    for m in MODELS:
        for kind in ["shift_speech", "shift_pause"]:
            Ks = sorted(int(c.split("_")[-1]) for (e, c, mm) in preds if e == "bench" and mm == m
                        and c.startswith(kind + "_"))
            pts, bs = zip(*[sc.boot(preds[("bench", f"{kind}_{k}", m)]) for k in Ks])
            pt, b = np.mean(pts, 0), np.mean(bs, 0)
            boots[("bench", f"{kind}_mean", m)] = (pt, b)
            lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
            for j, name in enumerate(CLASSES + ["macro"]):
                summ_rows.append({"experiment": "bench", "condition": f"{kind}_mean", "model": m,
                                  "metric": name, "point": round(float(pt[j]), 4),
                                  "lo": round(float(lo[j]), 4), "hi": round(float(hi[j]), 4),
                                  "fold_mean": "", "fold_sd": ""})

    # baselines
    base = {}
    base["corpus_only"] = lr_cv(corpus.reshape(-1, 1), y, folds, onehot=True)
    dur = np.array([[np.log(max(float(r["orig_dur_s"]), 1e-3)), int(r["used_samples"]) / SR,
                     float(r["speech_frac"]), int(r["n_speech_segments"]) / (int(r["used_samples"]) / SR)]
                    for r in meta])
    base["duration_vad_only"] = lr_cv(dur, y, folds)
    base["handcrafted_lr"] = lr_cv(handcrafted(work, idx, meta, args.jobs), y, folds)
    Pd = np.full((len(SEEDS), n), -1)
    for s in range(len(SEEDS)):
        for k in range(N_FOLDS):
            tr, te = folds[s] != k, folds[s] == k
            Pd[s, te] = DummyClassifier(strategy="stratified", random_state=SEEDS[s] * 100 + k
                                        ).fit(np.zeros((tr.sum(), 1)), y[tr]).predict(np.zeros((te.sum(), 1)))
    base["dummy_stratified"] = Pd
    for b, P in base.items():
        add("baseline", b, "-", P)
    prev = np.bincount(y, minlength=C) / n
    chance = {"uniform_random": 2 * prev * (1 / C) / (prev + 1 / C),
              "prior_matched_random": prev, "always_this_class": 2 * prev / (1 + prev)}
    for k, v in chance.items():
        for j, name in enumerate(CLASSES):
            summ_rows.append({"experiment": "chance_analytic", "condition": k, "model": "-",
                              "metric": name, "point": round(float(v[j]), 4)})
        if k != "always_this_class":
            summ_rows.append({"experiment": "chance_analytic", "condition": k, "model": "-",
                              "metric": "macro", "point": round(float(v.mean()), 4)})

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "condition", "model", "metric",
                                          "point", "lo", "hi", "fold_mean", "fold_sd"])
        w.writeheader()
        w.writerows(summ_rows)

    # paired comparisons
    pr = []

    def pair(tag, kA, kB):
        (pA, bA), (pB, bB) = boots[kA], boots[kB]
        d, lo, hi, p = Scorer.paired(bA, bB, pA, pB)
        for j, name in enumerate(CLASSES + ["macro"]):
            pr.append({"family": tag, "A": "|".join(kA), "B": "|".join(kB), "metric": name,
                       "diff": round(float(d[j]), 4), "lo": round(float(lo[j]), 4),
                       "hi": round(float(hi[j]), 4), "p": float(p[j])})

    for a, b in itertools.combinations(MODELS, 2):
        pair("models", ("bench", "full_valid", a), ("bench", "full_valid", b))
    for m in MODELS:
        pair("mask_trainfull", ("bench", "speech", m), ("bench", "pause", m))
        pair("mask_trainfull_vs_shift", ("bench", "speech", m), ("bench", "shift_speech_mean", m))
        pair("mask_trainfull_vs_shift", ("bench", "pause", m), ("bench", "shift_pause_mean", m))
        pair("mask_traintest", ("tt_speech", "speech", m), ("tt_pause", "pause", m))
        pair("mask_traintest_vs_shift", ("tt_speech", "speech", m), ("tt_shift_speech", "shift_speech_0", m))
        pair("mask_traintest_vs_shift", ("tt_pause", "pause", m), ("tt_shift_pause", "shift_pause_0", m))
        pair("reversal_trainfwd", ("bench", "full_valid", m), ("bench", "rev_valid", m))
        pair("local_reversal_trainfwd", ("bench", "full_valid", m), ("bench", "lrev_valid", m))
        pair("reversal_traintest", ("bench", "full_valid", m), ("tt_rev_valid", "rev_valid", m))
        pair("local_reversal_traintest", ("bench", "full_valid", m), ("tt_lrev_valid", "lrev_valid", m))
        if m in WHISPER:
            pair("whisper_pooling", ("bench_all", "full_all", m), ("bench", "full_valid", m))
    for b in base:
        for m in MODELS:
            pair("model_vs_baseline", ("bench", "full_valid", m), ("baseline", b, "-"))
    # Holm within each family x metric
    fam = defaultdict(list)
    for i, r in enumerate(pr):
        fam[(r["family"], r["metric"])].append(i)
    for ids in fam.values():
        for i, a in zip(ids, holm([pr[i]["p"] for i in ids])):
            pr[i]["p_holm"] = round(float(a), 4)
            pr[i]["p"] = round(pr[i]["p"], 4)
    with open(out / "paired.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "A", "B", "metric", "diff", "lo", "hi", "p", "p_holm"])
        w.writeheader()
        w.writerows(pr)

    # confusion matrices (pooled over seeds) for every model
    cms = {}
    for m in MODELS:
        P = preds[("bench", "full_valid", m)]
        cm = sum(confusion_matrix(y, p, labels=range(C)) for p in P)
        cms[m] = (cm / cm.sum(1, keepdims=True)).round(3).tolist()
    (out / "confusion.json").write_text(json.dumps(cms, indent=1))

    # corpus identification from embeddings
    corp_rows = []
    cl = np.unique(corpus, return_inverse=True)[1]
    cfolds = np.full((len(SEEDS), n), -1)
    for s, seed in enumerate(SEEDS):
        for k, (_, te) in enumerate(StratifiedGroupKFold(N_FOLDS, shuffle=True,
                                                          random_state=seed).split(cl, cl, groups)):
            cfolds[s, te] = k
    nco = cl.max() + 1
    for m in MODELS:
        P = lr_cv(feats[m]["full_valid"], cl, cfolds, nc=nco)
        f = np.mean([f1_score(cl, p, average="macro") for p in P])
        acc = np.mean([(p == cl).mean() for p in P])
        corp_rows.append({"model": m, "corpus_macro_f1": round(float(f), 4), "corpus_acc": round(float(acc), 4)})
    with open(out / "corpus_identification.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "corpus_macro_f1", "corpus_acc"])
        w.writeheader()
        w.writerows(corp_rows)

    # within-corpus analyses
    wrows = []
    for c in sorted(set(corpus)):
        sel = corpus == c
        present = [i for i in range(C)
                   if len(set(groups[sel & (y == i)])) >= args.min_within]
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
        for m in MODELS:
            X = feats[m]["full_valid"][sel]
            P = np.full((len(SEEDS), sel.sum()), -1)
            for s, seed in enumerate(SEEDS):
                outs = Parallel(n_jobs=args.jobs)(
                    delayed(fit_predict)(X[fl[s] != k], yc[fl[s] != k], [X[fl[s] == k]],
                                         len(present), seed * 100 + k) for k in range(k_splits))
                for k, o in enumerate(outs):
                    P[s, fl[s] == k] = o[0]
            pt, b = scw.boot(P)
            lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
            names = [CLASSES[i] for i in present] + ["macro"]
            for j, name in enumerate(names):
                wrows.append({"corpus": c, "classes": "/".join(CLASSES[i] for i in present),
                              "n": int(sel.sum()), "model": m, "metric": name,
                              "point": round(float(pt[j]), 4), "lo": round(float(lo[j]), 4),
                              "hi": round(float(hi[j]), 4)})
    with open(out / "within_corpus.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["corpus", "classes", "n", "model", "metric", "point", "lo", "hi"])
        w.writeheader()
        w.writerows(wrows)

    (out / "run_info.json").write_text(json.dumps(info, indent=1))
    print(f"Done. Results in {out}")


if __name__ == "__main__":
    main()
