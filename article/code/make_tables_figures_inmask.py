"""Figure, tables and prose numbers for the input-level region masking (revision R1, round D).

All numbers come from rerun/work/results_inmask (evaluate_inputmask.py, supp_inputmask.py)
and, for the frame-level series, from rerun/work/results_v2. Nothing is typed by hand.

Outputs: <fig>/fig_masking_diff.pdf (+ .png preview), <gen>/table_inmask.tex,
<gen>/table_inmask_within.tex, <gen>/numbers_inmask.json.

Usage:
  python make_assets_inmask.py --res results_inmask --fig <dir> --gen <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True, help="work directory that holds results_v2 and results_inmask")
ap.add_argument("--res", default="results_inmask")
ap.add_argument("--fig", required=True)
ap.add_argument("--gen", required=True)
args = ap.parse_args()
W = Path(args.work)
FIG, GEN = Path(args.fig), Path(args.gen)
FIG.mkdir(parents=True, exist_ok=True)
GEN.mkdir(parents=True, exist_ok=True)

s = pd.read_csv(W / args.res / "summary.csv")
p = pd.read_csv(W / args.res / "paired.csv")
wc = pd.read_csv(W / args.res / "within_corpus.csv")
wp = pd.read_csv(W / args.res / "within_paired.csv")
cid = pd.read_csv(W / args.res / "corpus_identification_inmask.csv")
gap = pd.read_csv(W / args.res / "gap_between_models.csv")
s2 = pd.read_csv(W / "results_v2" / "summary.csv")
p2 = pd.read_csv(W / "results_v2" / "paired.csv")
HAS_SHIFT = bool((s.condition == "in_shift_speech").any())

ORDER = ["wav2vec2_ssl", "wav2vec2", "hubert", "wavlm", "whisper_base", "whisper_medium", "xlsr", "unispeech_sat"]
NAME = {"wav2vec2_ssl": "wav2vec2-base", "wav2vec2": "wav2vec2-base-960h", "hubert": "HuBERT",
        "wavlm": "WavLM", "whisper_base": "Whisper-base", "whisper_medium": "Whisper-medium",
        "xlsr": "XLS-R", "unispeech_sat": "UniSpeech-SAT"}
CLS = ["Control", "MCI", "AD", "ProbableAD", "PPA"]
CORPORA = ["Pitt", "WLS", "Ivanova", "VAS", "Delaware", "Lu"]
ALPHA = 0.05


def S(df, exp, cond, model, metric):
    r = df[(df.experiment == exp) & (df.condition == cond) & (df.model == model) & (df.metric == metric)]
    assert len(r) == 1, (exp, cond, model, metric, len(r))
    return r.iloc[0]


def P(df, fam, A, B, metric):
    r = df[(df.family == fam) & (df.A == A) & (df.B == B) & (df.metric == metric)]
    assert len(r) == 1, (fam, A, B, metric, len(r))
    r = r.iloc[0]
    return float(r["diff"]), float(r.lo), float(r.hi), float(r.p_holm)


def f3(x):
    x = float(x)
    if abs(x) < 0.0005:
        x = 0.0
    return f"{x:.3f}".replace("-", "$-$")


def ci(pt, lo, hi):
    return r"\ci{" + f3(pt) + "}{" + f3(lo) + "}{" + f3(hi) + "}"


def tt(k):
    return "tt_" + k, k


def sp_pair(m, met, suf="", design="traintest"):
    a, b = "in_speech" + suf, "in_pause" + suf
    if design == "traintest":
        return P(p, "inmask_traintest" + suf, f"tt_{a}|{a}|{m}", f"tt_{b}|{b}|{m}", met)
    return P(p, "inmask_trainfull" + suf, f"bench|{a}|{m}", f"bench|{b}|{m}", met)


# ----------------------------------------------------------------------------- Table: pooled task
rows = []
for m in ORDER:
    for j, met in enumerate(["MCI", "AD", "macro"]):
        first = r"\multirow{3}{*}{" + NAME[m] + "}" if j == 0 else ""
        cells = [first, met if met != "macro" else "Macro", f3(S(s, "bench", "full_valid", m, met).point)]
        for k in ["in_speech", "in_pause"]:
            r = S(s, *tt(k), m, met)
            cells.append(ci(r.point, r.lo, r.hi))
        d, lo, hi, pv = sp_pair(m, met)
        cells.append(f3(d) + ("*" if pv < ALPHA else ""))
        if HAS_SHIFT:
            cells += [f3(S(s, *tt(k), m, met).point) for k in ["in_shift_speech", "in_shift_pause"]]
        cells += [f3(S(s, *tt(k), m, met).point) for k in ["in_speech_all", "in_pause_all"]]
        cells += [f3(S(s, "bench", k, m, met).point) for k in ["in_speech", "in_pause"]]
        rows.append(" & ".join(cells) + r" \\")
    if m != ORDER[-1]:
        rows.append(r"\midrule")
n1 = 5 if HAS_SHIFT else 3
shift_head = r" & \textbf{Shifted speech} & \textbf{Shifted pause}" if HAS_SHIFT else ""
shift_cap = (r" Shifted masks are the same masks circularly shifted by 25\% to 75\% of the recording, so they keep "
             r"the same amount of audio at the wrong positions.") if HAS_SHIFT else ""
t_in = r"""\begin{table}[!htbp]
  \ifrevisionmarkup\color{red}\fi
  \caption{\add{Input-level region masking results. Speech segments or pauses are replaced by digital silence in the waveform before encoding. Full gives F1 with the full representation. In the first block, separate classifiers are trained and tested on the speech-only and on the pause-only input, pooled over the frames of the retained region. Diff.\ is speech-only minus pause-only F1, and an asterisk marks a significant difference after Holm adjustment.""" + shift_cap + r""" The second block repeats the first with pooling over all frames. In the third block, classifiers trained on full representations are tested on the masked inputs. The 95\% CIs are given below the speech-only and pause-only values of the first block.}\label{tab:inmask}}
  \begin{adjustwidth}{-\extralength}{0cm}
  \scriptsize\ifrevisionmarkup\color{red}\fi
  \begin{tabularx}{\fulllength}{@{}>{\raggedright\arraybackslash}p{2.2cm}lc""" + "C" * (n1 + 4) + r"""@{}}
    \toprule
    & & & \multicolumn{""" + str(n1) + r"""}{c}{\textbf{Trained and tested on masked input}} & \multicolumn{2}{c}{\textbf{Pooled over all frames}} & \multicolumn{2}{c}{\textbf{Trained on full}} \\
    \cmidrule(lr){4-""" + str(3 + n1) + r"""}\cmidrule(lr){""" + str(4 + n1) + "-" + str(5 + n1) + r"""}\cmidrule(l){""" + str(6 + n1) + "-" + str(7 + n1) + r"""}
    \textbf{Model} & \textbf{Class} & \textbf{Full} & \textbf{Speech} & \textbf{Pause} & \textbf{Diff.}""" + shift_head + r""" & \textbf{Speech} & \textbf{Pause} & \textbf{Speech} & \textbf{Pause} \\
    \midrule
""" + "\n".join("    " + r for r in rows) + r"""
    \bottomrule
  \end{tabularx}
  \end{adjustwidth}
\end{table}
"""
(GEN / "table_inmask.tex").write_text(t_in)

# ----------------------------------------------------------------------------- Table: within corpus (macro-F1)
def W1(c, exp, cond, m, met="macro"):
    r = wc[(wc.corpus == c) & (wc.experiment == exp) & (wc.condition == cond) & (wc.model == m) & (wc.metric == met)]
    assert len(r) == 1, (c, exp, cond, m, met, len(r))
    return r.iloc[0]


def WP(c, m, met="macro", suf=""):
    a, b = "in_speech" + suf, "in_pause" + suf
    r = wp[(wp.corpus == c) & (wp.family == "inmask_traintest" + suf) & (wp.A == f"tt_{a}|{a}|{m}")
           & (wp.B == f"tt_{b}|{b}|{m}") & (wp.metric == met)]
    assert len(r) == 1, (c, m, met, suf, len(r))
    r = r.iloc[0]
    return float(r["diff"]), float(r.lo), float(r.hi), float(r.p_holm)


K = {c: len(wc[wc.corpus == c].classes.iloc[0].split("/")) for c in CORPORA}
rows = []
for m in ORDER:
    cells = [NAME[m]]
    for c in CORPORA:
        a, b = W1(c, *tt("in_speech"), m), W1(c, *tt("in_pause"), m)
        d, lo, hi, pv = WP(c, m)
        cells.append(r"\begin{tabular}[t]{@{}c@{}}" + f3(a.point) + " / " + f3(b.point) + r"\\[-1pt]{\scriptsize " + f3(d)
                     + ("*" if pv < ALPHA else "") + r"}\end{tabular}")
    rows.append(" & ".join(cells) + r" \\")
head = " & ".join([r"\textbf{Model}"] + [r"\textbf{" + c + "}" for c in CORPORA]) + r" \\"
sub = " & ".join([""] + ["/".join({"Control": "C", "MCI": "M", "AD": "A", "ProbableAD": "P"}[x]
                                  for x in wc[wc.corpus == c].classes.iloc[0].split("/"))
                         + f", $n$={int(wc[wc.corpus == c].n.iloc[0])}" for c in CORPORA]) + r" \\"
chance = " & ".join(["Chance"] + [f3(1 / K[c]) for c in CORPORA]) + r" \\"
t_w = r"""\begin{table}[!htbp]
  \ifrevisionmarkup\color{red}\fi
  \caption{\add{Within-corpus macro-F1 with input-level masking. Each cell gives the macro-F1 of classifiers trained and tested on speech-only input and on pause-only input (speech / pause, pooled over the frames of the retained region) and, below, their difference. An asterisk marks a significant difference after Holm adjustment within the corpus. C, M, A and P denote Control, MCI, AD and ProbableAD, and $n$ is the number of recordings.}\label{tab:inmaskwithin}}
  \begin{adjustwidth}{-\extralength}{0cm}
  \footnotesize\ifrevisionmarkup\color{red}\fi
  \begin{tabularx}{\fulllength}{@{}>{\raggedright\arraybackslash}p{3.0cm}CCCCCC@{}}
    \toprule
    """ + head + "\n    " + sub + r"""
    \midrule
""" + "\n".join("    " + r for r in rows) + r"""
    \midrule
    """ + chance + r"""
    \bottomrule
  \end{tabularx}
  \end{adjustwidth}
\end{table}
"""
(GEN / "table_inmask_within.tex").write_text(t_w)

# ----------------------------------------------------------------------------- numbers for the prose
def rng(v):
    return [round(float(min(v)), 3), round(float(max(v)), 3)]


num = {"has_shift": HAS_SHIFT, "models": [NAME[m] for m in ORDER]}
num["chance_macro"] = round(float(S(s2, "chance_analytic", "prior_matched_random", "-", "macro").point), 3)
num["frame_pause_macro"] = rng([S(s2, "tt_pause", "pause", m, "macro").point for m in ORDER])
num["frame_speech_macro"] = rng([S(s2, "tt_speech", "speech", m, "macro").point for m in ORDER])
for suf, tag in [("", "ret"), ("_all", "all")]:
    d = {}
    d["speech_macro"] = rng([S(s, *tt("in_speech" + suf), m, "macro").point for m in ORDER])
    d["pause_macro"] = rng([S(s, *tt("in_pause" + suf), m, "macro").point for m in ORDER])
    fs = [P(p, "inmask_traintest_vs_full" + suf, f"bench|full_valid|{m}", "tt_in_speech{0}|in_speech{0}|{1}".format(suf, m), "macro") for m in ORDER]
    fp = [P(p, "inmask_traintest_vs_full" + suf, f"bench|full_valid|{m}", "tt_in_pause{0}|in_pause{0}|{1}".format(suf, m), "macro") for m in ORDER]
    d["full_minus_speech_maxabs"] = round(max(abs(x[0]) for x in fs), 3)
    d["full_minus_speech_sig"] = [NAME[m] for m, x in zip(ORDER, fs) if x[3] < ALPHA]
    d["full_minus_pause_sig"] = [NAME[m] for m, x in zip(ORDER, fp) if x[3] < ALPHA]
    d["full_minus_pause_sig_range"] = rng([x[0] for x in fp if x[3] < ALPHA])
    for met in ["macro"] + CLS:
        pr = {m: sp_pair(m, met, suf) for m in ORDER}
        d[f"sp_{met}_sig_pos"] = [NAME[m] for m in ORDER if pr[m][3] < ALPHA and pr[m][0] > 0]
        d[f"sp_{met}_sig_neg"] = [NAME[m] for m in ORDER if pr[m][3] < ALPHA and pr[m][0] < 0]
        d[f"sp_{met}_sig_range"] = rng([pr[m][0] for m in ORDER if pr[m][3] < ALPHA]) if any(pr[m][3] < ALPHA for m in ORDER) else None
        d[f"sp_{met}_nonsig"] = {NAME[m]: [round(x, 3) for x in pr[m][:3]] for m in ORDER if pr[m][3] >= ALPHA}
    d["trainfull_speech_macro"] = rng([S(s, "bench", "in_speech" + suf, m, "macro").point for m in ORDER])
    d["trainfull_pause_macro"] = rng([S(s, "bench", "in_pause" + suf, m, "macro").point for m in ORDER])
    tf = {m: sp_pair(m, "macro", suf, "trainfull") for m in ORDER}
    d["trainfull_sp_macro_sig_pos"] = [NAME[m] for m in ORDER if tf[m][3] < ALPHA and tf[m][0] > 0]
    d["trainfull_sp_macro_sig_neg"] = [NAME[m] for m in ORDER if tf[m][3] < ALPHA and tf[m][0] < 0]
    for met in ["MCI", "AD"]:
        tfc = {m: sp_pair(m, met, suf, "trainfull") for m in ORDER}
        d[f"trainfull_sp_{met}_sig_neg"] = {NAME[m]: [round(x, 3) for x in tfc[m][:3]] for m in ORDER
                                            if tfc[m][3] < ALPHA and tfc[m][0] < 0}
    # corpus identification
    cs = cid[cid.input == "in_speech" + suf].set_index("model").loc[ORDER].corpus_acc
    cp = cid[cid.input == "in_pause" + suf].set_index("model").loc[ORDER].corpus_acc
    pm = pd.Series([S(s, *tt("in_pause" + suf), m, "macro").point for m in ORDER], index=ORDER)
    d["cid_speech"], d["cid_pause"] = rng(cs), rng(cp)
    d["cid_pause_top3"] = [NAME[m] for m in cp.sort_values(ascending=False).index[:3]]
    d["pause_macro_top3"] = [NAME[m] for m in pm.sort_values(ascending=False).index[:3]]
    d["cid_pause_vs_macro_spearman"] = round(float(cp.rank().corr(pm.rank())), 3)
    # within corpus
    wd = {}
    for c in CORPORA:
        pr = {m: WP(c, m, "macro", suf) for m in ORDER}
        lo_p = {m: float(W1(c, *tt("in_pause" + suf), m).lo) for m in ORDER}
        lo_s = {m: float(W1(c, *tt("in_speech" + suf), m).lo) for m in ORDER}
        wd[c] = {"chance": round(1 / K[c], 3),
                 "sig_pos": [NAME[m] for m in ORDER if pr[m][3] < ALPHA and pr[m][0] > 0],
                 "sig_neg": [NAME[m] for m in ORDER if pr[m][3] < ALPHA and pr[m][0] < 0],
                 "sig_range": rng([pr[m][0] for m in ORDER if pr[m][3] < ALPHA]) if any(pr[m][3] < ALPHA for m in ORDER) else None,
                 "diff_range": rng([pr[m][0] for m in ORDER]),
                 "sig_diffs": {NAME[m]: round(pr[m][0], 3) for m in ORDER if pr[m][3] < ALPHA},
                 "speech_macro": rng([W1(c, *tt("in_speech" + suf), m).point for m in ORDER]),
                 "pause_macro": rng([W1(c, *tt("in_pause" + suf), m).point for m in ORDER]),
                 "speech_mean": round(float(np.mean([W1(c, *tt("in_speech" + suf), m).point for m in ORDER])), 3),
                 "pause_mean": round(float(np.mean([W1(c, *tt("in_pause" + suf), m).point for m in ORDER])), 3),
                 "pause_above_chance": [NAME[m] for m in ORDER if lo_p[m] > 1 / K[c]],
                 "speech_above_chance": [NAME[m] for m in ORDER if lo_s[m] > 1 / K[c]]}
    d["within"] = wd
    d["within_n_combos"] = len(CORPORA) * len(ORDER)
    d["within_n_sig_pos"] = sum(len(v["sig_pos"]) for v in wd.values())
    d["within_n_sig_neg"] = sum(len(v["sig_neg"]) for v in wd.values())
    # gap between models
    g = gap[gap.pooling == ("retained" if suf == "" else "all")]
    d["gap_pairs"] = {met: [[NAME[r["A"]], NAME[r["B"]], round(float(r["diff"]), 3)]
                            for _, r in g[(g.metric == met) & (g.p_holm < ALPHA)].iterrows()]
                      for met in ["macro", "MCI", "AD"]}
    d["gap_n_pairs"] = int((g.metric == "macro").sum())
    if HAS_SHIFT:
        sh = {}
        for reg in ["speech", "pause"]:
            k, ks = f"in_{reg}{suf}", f"in_shift_{reg}{suf}"
            pr = {m: P(p, "inmask_traintest_vs_shift" + suf, f"tt_{k}|{k}|{m}", f"tt_{ks}|{ks}|{m}", "macro") for m in ORDER}
            sh[f"{reg}_shift_macro"] = rng([S(s, *tt(ks), m, "macro").point for m in ORDER])
            sh[f"{reg}_minus_shift_sig_pos"] = [NAME[m] for m in ORDER if pr[m][3] < ALPHA and pr[m][0] > 0]
            sh[f"{reg}_minus_shift_sig_neg"] = [NAME[m] for m in ORDER if pr[m][3] < ALPHA and pr[m][0] < 0]
            sh[f"{reg}_minus_shift_range"] = rng([pr[m][0] for m in ORDER])
            sh[f"{reg}_minus_shift"] = {NAME[m]: [round(x, 3) for x in pr[m]] for m in ORDER}
            sh[f"{reg}_minus_shift_n_pos"] = sum(pr[m][0] > 0 for m in ORDER)
            # every class: significant differences in either direction, with CI
            cls_pos, cls_neg = {}, {}
            for met in CLS:
                prc = {m: P(p, "inmask_traintest_vs_shift" + suf, f"tt_{k}|{k}|{m}", f"tt_{ks}|{ks}|{m}", met) for m in ORDER}
                for m in ORDER:
                    if prc[m][3] < ALPHA:
                        (cls_pos if prc[m][0] > 0 else cls_neg)[f"{NAME[m]}|{met}"] = [round(x, 3) for x in prc[m][:3]]
            sh[f"{reg}_minus_shift_class_sig_pos"], sh[f"{reg}_minus_shift_class_sig_neg"] = cls_pos, cls_neg
            sh[f"{reg}_minus_shift_class_all"] = {
                f"{NAME[m]}|{met}": [round(x, 3) for x in P(p, "inmask_traintest_vs_shift" + suf, f"tt_{k}|{k}|{m}", f"tt_{ks}|{ks}|{m}", met)]
                for m in ORDER for met in ["MCI", "AD"]}
            # within corpus, macro-F1
            wsh = {}
            for c in CORPORA:
                g = wp[(wp.corpus == c) & (wp.family == "inmask_traintest_vs_shift" + suf) & (wp.metric == "macro")]
                g = g[g.A.str.split("|").str[1] == k]
                assert len(g) == len(ORDER), (c, k, len(g))
                wsh[c] = {"sig_pos": [NAME[r["A"].split("|")[2]] for _, r in g.iterrows() if r.p_holm < ALPHA and r["diff"] > 0],
                          "sig_neg": [NAME[r["A"].split("|")[2]] for _, r in g.iterrows() if r.p_holm < ALPHA and r["diff"] < 0],
                          "n_neg": int((g["diff"] < 0).sum()), "range": rng(g["diff"])}
            sh[f"{reg}_minus_shift_within"] = wsh
        d["shift"] = sh
    num[tag] = d
(GEN / "numbers_inmask.json").write_text(json.dumps(num, indent=1))

# ----------------------------------------------------------------------------- figure
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK, "xtick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 0, "axes.titleweight": "bold",
})
y = np.arange(len(ORDER))[::-1].astype(float)


def errpt(ax, x, lo, hi, yy, color, marker):
    ax.plot([lo, hi], [yy, yy], color=color, lw=1.2, solid_capstyle="round", zorder=2)
    ax.plot([x], [yy], marker, color=color, ms=4.2, mec="white", mew=0.6, zorder=3)


SERIES = [("Frame-level masking, trained on full vectors, tested on masked vectors",
           p2, "mask_trainfull", "bench|speech|{m}", "bench|pause|{m}", 0.24, ORANGE, "s"),
          ("Frame-level masking, trained and tested on masked vectors",
           p2, "mask_traintest", "tt_speech|speech|{m}", "tt_pause|pause|{m}", 0.0, BLUE, "o"),
          ("Input-level masking, trained and tested on masked input",
           p, "inmask_traintest", "tt_in_speech|in_speech|{m}", "tt_in_pause|in_pause|{m}", -0.24, AQUA, "^")]
fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.2), sharey=True, layout="constrained")
ext = 0
for ax, met in zip(axes, ["MCI", "AD"]):
    ax.axvline(0, color=AXIS, lw=0.9, zorder=1)
    for yi, m in zip(y, ORDER):
        for lab, df, fam, A, B, off, col, mk in SERIES:
            d, lo, hi, pv = P(df, fam, A.format(m=m), B.format(m=m), met)
            errpt(ax, d, lo, hi, yi + off, col, mk)
            ext = max(ext, abs(lo), abs(hi))
    ax.set_title(met)
    for sp_ in ["top", "right", "left"]:
        ax.spines[sp_].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.7, len(ORDER) - 0.3)
for ax in axes:
    ax.set_xlim(-ext * 1.08, ext * 1.08)
axes[0].set_yticks(y)
axes[0].set_yticklabels([NAME[m] for m in ORDER])
fig.supxlabel("Speech-only F1 minus pause-only F1 (95% CI). Positive values: higher F1 with speech-only", fontsize=7.5)
h = [Line2D([0], [0], color=c[6], lw=1.2, marker=c[7], ms=4.2, mec="white", mew=0.6) for c in SERIES]
fig.legend(h, [c[0] for c in SERIES], loc="outside upper center", ncol=1, frameon=False)
fig.savefig(FIG / "fig_masking_diff.pdf")
fig.savefig(GEN / "fig_masking_diff_preview.png", dpi=200)
plt.close(fig)
print("written:", FIG / "fig_masking_diff.pdf", "and tables / numbers in", GEN)
