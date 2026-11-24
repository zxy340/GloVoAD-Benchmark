"""
Revision rerun, step 2c: text report of the input-level region masking.

Reads the CSV files written by evaluate_inputmask.py (and, for comparison, the
frame-level masking results of evaluate_v2.py) and prints every number that is
quoted in the manuscript. Nothing is computed here except counts and ranges.

Usage:
  python report_inputmask.py --work work/                      # results_inmask
  python report_inputmask.py --work work/ --res results_inmask_main
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

MODELS = ["wav2vec2_ssl", "wav2vec2", "hubert", "wavlm", "whisper_base", "whisper_medium", "xlsr", "unispeech_sat"]
NAME = {"wav2vec2_ssl": "wav2vec2-base", "wav2vec2": "wav2vec2-960h", "hubert": "HuBERT", "wavlm": "WavLM",
        "whisper_base": "Whisper-base", "whisper_medium": "Whisper-medium", "xlsr": "XLS-R",
        "unispeech_sat": "UniSpeech-SAT"}
CLASSES = ["Control", "MCI", "AD", "ProbableAD", "PPA"]


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fmt(pt, lo, hi):
    return f"{float(pt):.3f} [{float(lo):.3f}, {float(hi):.3f}]"


def sig(r):
    return float(r["p_holm"]) < 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--res", default="results_inmask")
    args = ap.parse_args()
    work = Path(args.work)
    res = work / args.res

    S = {(r["experiment"], r["condition"], r["model"], r["metric"]): r for r in read(res / "summary.csv")}
    P = {(r["family"], r["A"], r["B"], r["metric"]): r for r in read(res / "paired.csv")}
    S2 = {(r["experiment"], r["condition"], r["model"], r["metric"]): r
          for r in read(work / "results_v2" / "summary.csv")}
    P2 = {(r["family"], r["A"], r["B"], r["metric"]): r for r in read(work / "results_v2" / "paired.csv")}
    has_shift = any(k[1] == "in_shift_speech" for k in S)
    global MODELS
    MODELS = [m for m in MODELS if ("bench", "full_valid", m, "macro") in S]

    def s(exp, cond, m, met):
        r = S[(exp, cond, m, met)]
        return fmt(r["point"], r["lo"], r["hi"])

    def pair(fam, A, B, met):
        r = P[(fam, "|".join(A), "|".join(B), met)]
        return r, f"{float(r['diff']):+.3f} [{float(r['lo']):+.3f}, {float(r['hi']):+.3f}] p={float(r['p_holm']):.3f}" \
            + ("*" if sig(r) else " ")

    for suf, title in [("", "PRIMARY pooling: mean over the frames of the retained region"),
                       ("_all", "SENSITIVITY pooling: mean over all frames")]:
        sp, pa = "in_speech" + suf, "in_pause" + suf
        print("=" * 110)
        print(f"POOLED FIVE-CLASS TASK, input-level masking, {title}")
        print("=" * 110)
        for design, exp_s, exp_p, fam in [("trained and tested on the gated input", "tt_" + sp, "tt_" + pa,
                                           "inmask_traintest" + suf),
                                          ("trained on full, tested on the gated input", "bench", "bench",
                                           "inmask_trainfull" + suf)]:
            print(f"\n--- {design}")
            for met in ["macro", "MCI", "AD"]:
                print(f"\n  {met}")
                print(f"  {'model':15s} {'full':>7s}  {'speech-only':24s} {'pause-only':24s} speech - pause (Holm)")
                n_sig = 0
                for m in MODELS:
                    r, txt = pair(fam, (exp_s, sp, m), (exp_p, pa, m), met)
                    n_sig += sig(r)
                    print(f"  {NAME[m]:15s} {float(S[('bench', 'full_valid', m, met)]['point']):7.3f}  "
                          f"{s(exp_s, sp, m, met):24s} {s(exp_p, pa, m, met):24s} {txt}")
                print(f"  significant after Holm: {n_sig} of {len(MODELS)}")
            rows = [P[(fam, "|".join((exp_s, sp, m)), "|".join((exp_p, pa, m)), c)] for m in MODELS for c in CLASSES]
            by_class = defaultdict(list)
            for r, (m, c) in zip(rows, [(m, c) for m in MODELS for c in CLASSES]):
                if sig(r):
                    by_class[c].append(f"{NAME[m]} {float(r['diff']):+.3f}")
            print("\n  significant speech - pause differences by class:")
            for c in CLASSES:
                print(f"    {c:11s} {len(by_class[c])}: {', '.join(by_class[c])}")

        print("\n--- full representation minus gated input (matched training), macro-F1")
        fam = "inmask_traintest_vs_full" + suf
        for m in MODELS:
            _, a = pair(fam, ("bench", "full_valid", m), ("tt_" + sp, sp, m), "macro")
            _, b = pair(fam, ("bench", "full_valid", m), ("tt_" + pa, pa, m), "macro")
            print(f"  {NAME[m]:15s} full - speech {a}   full - pause {b}")
        for met in ["MCI", "AD"]:
            ns = sum(sig(P[(fam, "|".join(("bench", "full_valid", m)), "|".join(("tt_" + k, k, m)), met)])
                     for m in MODELS for k in (sp, pa))
            print(f"  {met}: {ns} of {2 * len(MODELS)} full-vs-gated differences significant")

        if has_shift:
            ss, sh = "in_shift_speech" + suf, "in_shift_pause" + suf
            for design, pre, fam in [("matched training", "tt_", "inmask_traintest_vs_shift" + suf),
                                     ("trained on full", "", "inmask_trainfull_vs_shift" + suf)]:
                print(f"\n--- VAD gate minus shifted gate ({design}), macro-F1")
                for m in MODELS:
                    ea = (lambda k: (pre + k if pre else "bench", k, m))
                    _, a = pair(fam, ea(sp), ea(ss), "macro")
                    _, b = pair(fam, ea(pa), ea(sh), "macro")
                    print(f"  {NAME[m]:15s} speech {s(*ea(ss), 'macro')} {a}   pause {s(*ea(sh), 'macro')} {b}")
                for met in ["MCI", "AD"]:
                    ns = sum(sig(P[(fam, "|".join(ea(a)), "|".join(ea(b)), met)])
                             for m in MODELS for ea in [lambda k, m=m: (pre + k if pre else "bench", k, m)]
                             for a, b in ((sp, ss), (pa, sh)))
                    print(f"  {met}: {ns} of {2 * len(MODELS)} gate-vs-shifted differences significant")

    print("\n" + "=" * 110)
    print("FOR COMPARISON: frame-level masking of the benchmark run (matched training)")
    print("=" * 110)
    for met in ["macro", "MCI", "AD"]:
        print(f"\n  {met}")
        for m in MODELS:
            r = P2[("mask_traintest", f"tt_speech|speech|{m}", f"tt_pause|pause|{m}", met)]
            a, b = S2[("tt_speech", "speech", m, met)], S2[("tt_pause", "pause", m, met)]
            print(f"  {NAME[m]:15s} speech {float(a['point']):.3f}  pause {float(b['point']):.3f}  "
                  f"diff {float(r['diff']):+.3f} [{float(r['lo']):+.3f}, {float(r['hi']):+.3f}] "
                  f"p={float(r['p_holm']):.3f}{'*' if sig(r) else ''}   "
                  f"input-level: speech {float(S[('tt_in_speech', 'in_speech', m, met)]['point']):.3f}  "
                  f"pause {float(S[('tt_in_pause', 'in_pause', m, met)]['point']):.3f}")

    # ------------------------------------------------------------------ within corpus
    W = read(res / "within_corpus.csv")
    WP = read(res / "within_paired.csv")
    Ws = {(r["corpus"], r["experiment"], r["condition"], r["model"], r["metric"]): r for r in W}
    Wp = {(r["corpus"], r["family"], r["A"], r["B"], r["metric"]): r for r in WP}
    corpora = sorted({r["corpus"] for r in W})
    for suf, title in [("", "PRIMARY pooling"), ("_all", "SENSITIVITY pooling")]:
        sp, pa = "in_speech" + suf, "in_pause" + suf
        print("\n" + "=" * 110)
        print(f"WITHIN-CORPUS CLASSIFICATION, matched training, {title}")
        print("=" * 110)
        tot = defaultdict(int)
        for c in corpora:
            cls = next(r["classes"] for r in W if r["corpus"] == c)
            n = next(r["n"] for r in W if r["corpus"] == c)
            print(f"\n--- {c} ({cls}, n={n})")
            print(f"  {'model':15s} {'full':24s} {'speech-only':24s} {'pause-only':24s} speech - pause (Holm)   "
                  f"full - speech / full - pause")
            for m in MODELS:
                f = Ws[(c, "bench", "full_valid", m, "macro")]
                a = Ws[(c, "tt_" + sp, sp, m, "macro")]
                b = Ws[(c, "tt_" + pa, pa, m, "macro")]
                r = Wp[(c, "inmask_traintest" + suf, f"tt_{sp}|{sp}|{m}", f"tt_{pa}|{pa}|{m}", "macro")]
                fa = Wp[(c, "inmask_traintest_vs_full" + suf, f"bench|full_valid|{m}", f"tt_{sp}|{sp}|{m}", "macro")]
                fb = Wp[(c, "inmask_traintest_vs_full" + suf, f"bench|full_valid|{m}", f"tt_{pa}|{pa}|{m}", "macro")]
                tot["n"] += 1
                tot["speech>pause"] += sig(r) and float(r["diff"]) > 0
                tot["pause>speech"] += sig(r) and float(r["diff"]) < 0
                tot["full>speech"] += sig(fa) and float(fa["diff"]) > 0
                tot["full>pause"] += sig(fb) and float(fb["diff"]) > 0
                tot["pos"] += float(r["diff"]) > 0
                print(f"  {NAME[m]:15s} {fmt(f['point'], f['lo'], f['hi']):24s} {fmt(a['point'], a['lo'], a['hi']):24s} "
                      f"{fmt(b['point'], b['lo'], b['hi']):24s} "
                      f"{float(r['diff']):+.3f} [{float(r['lo']):+.3f}, {float(r['hi']):+.3f}] "
                      f"p={float(r['p_holm']):.3f}{'*' if sig(r) else ' '}   "
                      f"{float(fa['diff']):+.3f}{'*' if sig(fa) else ' '} / {float(fb['diff']):+.3f}{'*' if sig(fb) else ' '}")
            per_class = [r for r in WP if r["corpus"] == c and r["family"] == "inmask_traintest" + suf
                         and r["metric"] != "macro" and sig(r)]
            if per_class:
                print("  significant per-class speech - pause differences: " + ", ".join(
                    f"{NAME[r['A'].split('|')[2]]} {r['metric']} {float(r['diff']):+.3f}" for r in per_class))
        print(f"\n  {tot['n']} corpus-by-model combinations (macro-F1): speech-only higher in {tot['pos']}, "
              f"significantly higher in {tot['speech>pause']}, pause-only significantly higher in {tot['pause>speech']}; "
              f"full significantly above speech-only in {tot['full>speech']}, above pause-only in {tot['full>pause']}")


if __name__ == "__main__":
    main()
