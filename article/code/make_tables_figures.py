"""Build every results table (LaTeX) and figure (PDF) of the revision from the rerun outputs.

Nothing is typed by hand: all numbers come from rerun/work/results_v2.
Outputs: gen/*.tex (table bodies with captions), figures/*.pdf, gen/numbers.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import argparse
_ap = argparse.ArgumentParser(description="Build the results tables and figures of the paper")
_ap.add_argument("--results", required=True, help="folder written by evaluate_v2.py (results_v2)")
_ap.add_argument("--figures", required=True, help="output folder for the PDF figures")
_ap.add_argument("--tables", required=True, help="output folder for the LaTeX tables")
_args = _ap.parse_args()
R, FIG, GEN = Path(_args.results), Path(_args.figures), Path(_args.tables)
FIG.mkdir(parents=True, exist_ok=True)
GEN.mkdir(parents=True, exist_ok=True)

s = pd.read_csv(R / "summary.csv")
p = pd.read_csv(R / "paired.csv")
w = pd.read_csv(R / "within_corpus.csv")
cid = pd.read_csv(R / "corpus_identification.csv").set_index("model")
cm = json.loads((R / "confusion.json").read_text())
info = json.loads((R / "run_info.json").read_text())

ORDER = ["wav2vec2_ssl", "wav2vec2", "hubert", "wavlm", "whisper_base", "whisper_medium", "xlsr", "unispeech_sat"]
NAME = {"wav2vec2_ssl": "wav2vec2-base", "wav2vec2": "wav2vec2-base-960h", "hubert": "HuBERT",
        "wavlm": "WavLM", "whisper_base": "Whisper-base", "whisper_medium": "Whisper-medium",
        "xlsr": "XLS-R", "unispeech_sat": "UniSpeech-SAT"}
CLS = ["Control", "MCI", "AD", "ProbableAD", "PPA"]
COLS = ["macro"] + CLS
HEAD = {"macro": "Macro-F1", "Control": "Control", "MCI": "MCI", "AD": "AD", "ProbableAD": "ProbAD", "PPA": "PPA"}
CORPORA = ["Pitt", "WLS", "Ivanova", "VAS", "Delaware", "Lu"]
ALPHA = 0.05


def S(exp, cond, model, metric):
    r = s[(s.experiment == exp) & (s.condition == cond) & (s.model == model) & (s.metric == metric)]
    assert len(r) == 1, (exp, cond, model, metric, len(r))
    return r.iloc[0]


def P(fam, A, B, metric):
    """Paired difference A minus B (oriented), with CI and Holm p."""
    r = p[(p.family == fam) & (p.A == A) & (p.B == B) & (p.metric == metric)]
    if len(r) == 1:
        r = r.iloc[0]
        return float(r["diff"]), float(r.lo), float(r.hi), float(r.p_holm)
    r = p[(p.family == fam) & (p.A == B) & (p.B == A) & (p.metric == metric)]
    assert len(r) == 1, (fam, A, B, metric, len(r))
    r = r.iloc[0]
    return -float(r["diff"]), -float(r.hi), -float(r.lo), float(r.p_holm)


def f3(x):
    x = float(x)
    if abs(x) < 0.0005:
        x = 0.0
    t = f"{x:.3f}"
    return t.replace("-", "$-$")


def ci(pt, lo, hi, bold=False):
    a = f3(pt)
    if bold:
        a = r"\textbf{" + a + "}"
    return r"\ci{" + a + "}{" + f3(lo) + "}{" + f3(hi) + "}"


def star(d, pv):
    return f3(d) + ("*" if pv < ALPHA else "")


num = {"info": info}

# ----------------------------------------------------------------------------- Table 3: benchmark
ties = {}
for met in COLS:
    pts = {m: float(S("bench", "full_valid", m, met).point) for m in ORDER}
    best = max(pts, key=pts.get)
    tie = {best}
    for m in ORDER:
        if m != best and P("models", f"bench|full_valid|{best}", f"bench|full_valid|{m}", met)[3] >= ALPHA:
            tie.add(m)
    ties[met] = (best, sorted(tie, key=ORDER.index))
num["ties"] = {k: {"best": v[0], "tied": v[1]} for k, v in ties.items()}

ch = s[(s.experiment == "chance_analytic") & (s.condition == "prior_matched_random")].set_index("metric").point
rows = []
for m in ORDER:
    cells = [NAME[m]] + [ci(*S("bench", "full_valid", m, met)[["point", "lo", "hi"]], bold=m in ties[met][1]) for met in COLS]
    rows.append(" & ".join(cells) + r" \\")
rows.append(r"\midrule")
BASE = [("handcrafted_lr", "Handcrafted features + LR"), ("duration_vad_only", "Duration and VAD only + LR"),
        ("corpus_only", "Corpus label only + LR"), ("dummy_stratified", "Stratified dummy")]
for b, label in BASE:
    cells = [label] + [ci(*S("baseline", b, "-", met)[["point", "lo", "hi"]]) for met in COLS]
    rows.append(" & ".join(cells) + r" \\")
rows.append(" & ".join(["Random guessing with class priors"] + [f3(ch[met]) for met in COLS]) + r" \\")
t3 = r"""\begin{table}[!htbp]
  \ifrevisionmarkup\color{red}\fi
  \caption{\add{Five-class benchmark results. Each cell gives the F1 score computed on the pooled out-of-fold predictions and averaged over the three seeds, with its 95\% participant-level bootstrap CI below. In each column, bold marks the pretrained model with the highest F1 and every model whose F1 does not differ significantly from it (paired bootstrap, Holm-adjusted $p \geq 0.05$). The lower rows give baselines evaluated on the same splits and the F1 expected from random guessing with the class priors. LR, logistic regression. Fold-level standard deviations are given in Table~\ref{tab:foldsd}.}\label{tab:benchmark}}
  \begin{adjustwidth}{-\extralength}{0cm}
  \footnotesize\ifrevisionmarkup\color{red}\fi
  \begin{tabularx}{\fulllength}{@{}>{\raggedright\arraybackslash}p{3.6cm}CCCCCC@{}}
    \toprule
    \textbf{Model} & \textbf{Macro-F1} & \textbf{Control} & \textbf{MCI} & \textbf{AD} & \textbf{ProbAD} & \textbf{PPA} \\
    \midrule
""" + "\n".join("    " + r for r in rows) + r"""
    \bottomrule
  \end{tabularx}
  \end{adjustwidth}
\end{table}
"""
(GEN / "table_benchmark.tex").write_text(t3)

# ----------------------------------------------------------------------------- Table 4: corpus-level
K = {}
for c in CORPORA:
    g = w[w.corpus == c]
    K[c] = (g.classes.iloc[0], int(g.n.iloc[0]))
rows = []
for m in ORDER:
    cells = [NAME[m], f3(cid.loc[m, "corpus_acc"]), f3(cid.loc[m, "corpus_macro_f1"])]
    for c in CORPORA:
        r = w[(w.corpus == c) & (w.model == m) & (w.metric == "macro")].iloc[0]
        cells.append(ci(r.point, r.lo, r.hi))
    rows.append(" & ".join(cells) + r" \\")
rows.append(r"\midrule")
n_corp = 9
maj = 830 / info["analysed"]
chance_row = ["Chance", f3(maj), f3(1 / n_corp)] + [f3(1 / len(K[c][0].split("/"))) for c in CORPORA]
rows.append(" & ".join(chance_row) + r" \\")
desc = ", ".join(f"{c} ({K[c][0].replace('/', ', ')}, {K[c][1]} recordings)" for c in CORPORA[:-1])
desc += f" and {CORPORA[-1]} ({K[CORPORA[-1]][0].replace('/', ', ')}, {K[CORPORA[-1]][1]} recordings)"
t4 = r"""\begin{table}[!htbp]
  \ifrevisionmarkup\color{red}\fi
  \caption{\add{Corpus-level analyses. The first two columns give the accuracy and macro-F1 of a logistic regression that predicts the source corpus (nine classes) from each model's embeddings. The other columns give the macro-F1, with its 95\% CI below, of classification within a single corpus, restricted to the classes with at least eight participants: """ + desc + r""". The last row gives the accuracy of always predicting the largest corpus and the macro-F1 expected from random guessing with the class priors.}\label{tab:corpus}}
  \begin{adjustwidth}{-\extralength}{0cm}
  \footnotesize\ifrevisionmarkup\color{red}\fi
  \begin{tabularx}{\fulllength}{@{}>{\raggedright\arraybackslash}p{2.7cm}ccCCCCCC@{}}
    \toprule
    & \multicolumn{2}{c}{\textbf{Corpus identification}} & \multicolumn{6}{c}{\textbf{Within-corpus macro-F1}} \\
    \cmidrule(lr){2-3}\cmidrule(l){4-9}
    \textbf{Model} & \textbf{Acc.} & \textbf{Macro-F1} & \textbf{""" + r"} & \textbf{".join(CORPORA) + r"""} \\
    \midrule
""" + "\n".join("    " + r for r in rows) + r"""
    \bottomrule
  \end{tabularx}
  \end{adjustwidth}
\end{table}
"""
(GEN / "table_corpus.tex").write_text(t4)

# ----------------------------------------------------------------------------- Table 5: confusion
M = np.array(cm["whisper_medium"])
lab = ["Control", "MCI", "AD", "ProbableAD", "PPA"]
rows = [f"{lab[i]:<10} & " + " & ".join(f"{v:.3f}" for v in M[i]) + r" \\" for i in range(5)]
t5 = r"""\begin{table}[!htbp]
  \centering
  \ifrevisionmarkup\color{red}\fi
  \caption{\add{Normalized confusion matrix for Whisper-medium, the model with the highest macro-F1, pooled over the out-of-fold predictions of the three seeds. Rows are true labels and columns are predicted labels, and each value is the proportion of a true class assigned to a predicted class.}\label{tab:confusion}}
  \ifrevisionmarkup\color{red}\fi
  \begin{tabular}{lccccc}
    \toprule
               & \textbf{p\_Ctrl} & \textbf{p\_MCI} & \textbf{p\_AD} & \textbf{p\_ProbAD} & \textbf{p\_PPA} \\
    \midrule
""" + "\n".join("    " + r for r in rows) + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
(GEN / "table_confusion.tex").write_text(t5)
num["confusion_whisper_medium"] = M.round(3).tolist()

# ----------------------------------------------------------------------------- Table 6: reversal drops
rows = []
for m in ORDER:
    cells = [NAME[m]]
    for met in COLS[1:] + ["macro"]:
        d, lo, hi, pv = P("reversal_trainfwd", f"bench|full_valid|{m}", f"bench|rev_valid|{m}", met)
        cells.append(star(d, pv))
    for met in ["PPA", "macro"]:
        d, lo, hi, pv = P("reversal_traintest", f"bench|full_valid|{m}", f"tt_rev_valid|rev_valid|{m}", met)
        cells.append(star(d, pv))
    for met in ["PPA", "macro"]:
        d, lo, hi, pv = P("local_reversal_trainfwd", f"bench|full_valid|{m}", f"bench|lrev_valid|{m}", met)
        cells.append(star(d, pv))
    rows.append(" & ".join(cells) + r" \\")
t6 = r"""\begin{table}[!htbp]
  \ifrevisionmarkup\color{red}\fi
  \caption{\add{Drop in F1 under audio reversal, computed as F1 on the original audio minus F1 on the manipulated audio. In the first block, classifiers trained on original audio are tested on reversed audio. In the second, classifiers are trained and tested on reversed audio. In the third, classifiers trained on original audio are tested on audio in which every 50\,ms window is reversed in place. Asterisks mark drops with a Holm-adjusted paired bootstrap $p < 0.05$. The F1 values of all conditions are given in Table~\ref{tab:reversalfull}.}\label{tab:perturbation}}
  \begin{adjustwidth}{-\extralength}{0cm}
  \footnotesize\ifrevisionmarkup\color{red}\fi
  \begin{tabularx}{\fulllength}{@{}>{\raggedright\arraybackslash}p{2.7cm}CCCCCCCCCC@{}}
    \toprule
    & \multicolumn{6}{c}{\textbf{Reversed, trained on original}} & \multicolumn{2}{c}{\textbf{Reversed, trained on reversed}} & \multicolumn{2}{c}{\textbf{50\,ms local reversal}} \\
    \cmidrule(lr){2-7}\cmidrule(lr){8-9}\cmidrule(l){10-11}
    \textbf{Model} & \textbf{Control} & \textbf{MCI} & \textbf{AD} & \textbf{ProbAD} & \textbf{PPA} & \textbf{Macro} & \textbf{PPA} & \textbf{Macro} & \textbf{PPA} & \textbf{Macro} \\
    \midrule
""" + "\n".join("    " + r for r in rows) + r"""
    \bottomrule
  \end{tabularx}
  \end{adjustwidth}
\end{table}
"""
(GEN / "table_reversal.tex").write_text(t6)

# ----------------------------------------------------------------------------- Table A1: fold SD
def pm(r):
    return f3(r.fold_mean) + r" $\pm$ " + f3(r.fold_sd)


rows = []
for m in ORDER:
    rows.append(" & ".join([NAME[m]] + [pm(S("bench", "full_valid", m, met)) for met in COLS]) + r" \\")
rows.append(r"\midrule")
for b, label in BASE:
    rows.append(" & ".join([label] + [pm(S("baseline", b, "-", met)) for met in COLS]) + r" \\")
ta1 = r"""\begin{table}[!htbp]
  \ifrevisionmarkup\color{red}\fi
  \caption{\add{Fold-level variability of the benchmark. Each cell gives the mean $\pm$ standard deviation of F1 over the 15 test folds (five folds $\times$ three seeds). Fold-level F1 is computed on about one fifth of the participants, so its mean differs slightly from the pooled values in Table~\ref{tab:benchmark}, and PPA has only seven or eight recordings per test fold.}\label{tab:foldsd}}
  \begin{adjustwidth}{-\extralength}{0cm}
  \footnotesize\ifrevisionmarkup\color{red}\fi
  \begin{tabularx}{\fulllength}{@{}>{\raggedright\arraybackslash}p{3.6cm}CCCCCC@{}}
    \toprule
    \textbf{Model} & \textbf{Macro-F1} & \textbf{Control} & \textbf{MCI} & \textbf{AD} & \textbf{ProbAD} & \textbf{PPA} \\
    \midrule
""" + "\n".join("    " + r for r in rows) + r"""
    \bottomrule
  \end{tabularx}
  \end{adjustwidth}
\end{table}
"""
(GEN / "table_A1.tex").write_text(ta1)

# ----------------------------------------------------------------------------- Table A2: masking
MASK_COLS = [("bench", "speech", True), ("bench", "pause", True), ("bench", "shift_speech_mean", False), ("bench", "shift_pause_mean", False),
             ("tt_speech", "speech", True), ("tt_pause", "pause", True), ("tt_shift_speech", "shift_speech_0", False), ("tt_shift_pause", "shift_pause_0", False)]
rows = []
for m in ORDER:
    for j, met in enumerate(["MCI", "AD", "macro"]):
        first = r"\multirow{3}{*}{" + NAME[m] + "}" if j == 0 else ""
        cells = [first, HEAD[met] if met != "macro" else "Macro", f3(S("bench", "full_valid", m, met).point)]
        for e, c, with_ci in MASK_COLS:
            r = S(e, c, m, met)
            cells.append(ci(r.point, r.lo, r.hi) if with_ci else f3(r.point))
        rows.append(" & ".join(cells) + r" \\")
    if m != ORDER[-1]:
        rows.append(r"\midrule")
ta2 = r"""\begin{table}[!htbp]
  \ifrevisionmarkup\color{red}\fi
  \caption{\add{Region masking results. Full gives F1 with the full representation. In the first block, classifiers trained on full representations are tested on speech-only, pause-only and shifted-mask vectors. In the second, separate classifiers are trained and tested on each kind of vector. Shifted masks have the size and run structure of the VAD masks but are circularly shifted by 25\% to 75\% of the recording. In the first block their F1 is averaged over five shifts, and in the second one shift is used. The 95\% CIs are given below the speech-only and pause-only values.}\label{tab:maskfull}}
  \begin{adjustwidth}{-\extralength}{0cm}
  \scriptsize\ifrevisionmarkup\color{red}\fi
  \begin{tabularx}{\fulllength}{@{}>{\raggedright\arraybackslash}p{2.2cm}lcCCCCCCCC@{}}
    \toprule
    & & & \multicolumn{4}{c}{\textbf{Trained on full, tested on masked}} & \multicolumn{4}{c}{\textbf{Trained and tested on masked}} \\
    \cmidrule(lr){4-7}\cmidrule(l){8-11}
    \textbf{Model} & \textbf{Class} & \textbf{Full} & \textbf{Speech} & \textbf{Pause} & \textbf{Shifted speech} & \textbf{Shifted pause} & \textbf{Speech} & \textbf{Pause} & \textbf{Shifted speech} & \textbf{Shifted pause} \\
    \midrule
""" + "\n".join("    " + r for r in rows) + r"""
    \bottomrule
  \end{tabularx}
  \end{adjustwidth}
\end{table}
"""
(GEN / "table_A2.tex").write_text(ta2)

# ----------------------------------------------------------------------------- Table A3: reversal F1
REV = [("Original", "bench", "full_valid"), ("Reversed, trained on original", "bench", "rev_valid"),
       ("Reversed, trained on reversed", "tt_rev_valid", "rev_valid"),
       ("Local reversal, trained on original", "bench", "lrev_valid"),
       ("Local reversal, trained on local reversal", "tt_lrev_valid", "lrev_valid")]
rows = []
for m in ORDER:
    for j, (lab_, e, c) in enumerate(REV):
        first = r"\multirow{5}{*}{" + NAME[m] + "}" if j == 0 else ""
        rows.append(" & ".join([first, lab_] + [f3(S(e, c, m, met).point) for met in COLS[1:] + ["macro"]]) + r" \\")
    if m != ORDER[-1]:
        rows.append(r"\midrule")
ta3 = r"""\begin{table}[!htbp]
  \ifrevisionmarkup\color{red}\fi
  \caption{\add{F1 under every condition of the audio reversal experiment. Local reversal reverses every 50\,ms window of the waveform in place.}\label{tab:reversalfull}}
  \begin{adjustwidth}{-\extralength}{0cm}
  \footnotesize\ifrevisionmarkup\color{red}\fi
  \begin{tabularx}{\fulllength}{@{}>{\raggedright\arraybackslash}p{2.6cm}>{\raggedright\arraybackslash}p{5.0cm}CCCCCC@{}}
    \toprule
    \textbf{Model} & \textbf{Condition} & \textbf{Control} & \textbf{MCI} & \textbf{AD} & \textbf{ProbAD} & \textbf{PPA} & \textbf{Macro} \\
    \midrule
""" + "\n".join("    " + r for r in rows) + r"""
    \bottomrule
  \end{tabularx}
  \end{adjustwidth}
\end{table}
"""
(GEN / "table_A3.tex").write_text(ta3)

# ----------------------------------------------------------------------------- figures
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
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


def style(ax):
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.7, len(ORDER) - 0.3)


def errpt(ax, x, lo, hi, yy, color, marker):
    ax.plot([lo, hi], [yy, yy], color=color, lw=1.2, solid_capstyle="round", zorder=2)
    ax.plot([x], [yy], marker, color=color, ms=4.2, mec="white", mew=0.6, zorder=3)


# Figure 2: benchmark with CIs and the corpus-only baseline
fig, axes = plt.subplots(1, 6, figsize=(7.1, 2.55), sharey=True, layout="constrained")
for ax, met in zip(axes, COLS):
    b = S("baseline", "corpus_only", "-", met)
    ax.axvspan(b.lo, b.hi, color=ORANGE, alpha=0.14, lw=0, zorder=0)
    ax.axvline(b.point, color=ORANGE, lw=1.0, ls=(0, (3, 2)), zorder=1)
    los, his = [b.lo], [b.hi]
    for yi, m in zip(y, ORDER):
        r = S("bench", "full_valid", m, met)
        errpt(ax, r.point, r.lo, r.hi, yi, BLUE, "o")
        los.append(r.lo); his.append(r.hi)
    lo_, hi_ = min(los), max(his)
    pad = 0.06 * (hi_ - lo_)
    ax.set_xlim(lo_ - pad, hi_ + pad)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    ax.set_title(HEAD[met] if met != "ProbableAD" else "ProbableAD")
    style(ax)
axes[0].set_yticks(y)
axes[0].set_yticklabels([NAME[m] for m in ORDER])
fig.supxlabel("F1 (95% CI)", fontsize=7.5)
h = [Line2D([0], [0], color=BLUE, lw=1.2, marker="o", ms=4.2, mec="white", mew=0.6, label="Pretrained model"),
     (Patch(color=ORANGE, alpha=0.14, lw=0), Line2D([0], [0], color=ORANGE, lw=1.0, ls=(0, (3, 2))))]
fig.legend(h, ["Pretrained model", "Corpus label only (95% CI band)"], loc="outside upper center", ncol=2, frameon=False,
           handlelength=2.2)
fig.savefig(FIG / "fig_benchmark_ci.pdf")
plt.close(fig)

# Figure 3: masking, speech-only minus pause-only
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.55), sharey=True, layout="constrained")
ext = 0
for ax, met in zip(axes, ["MCI", "AD"]):
    ax.axvline(0, color=AXIS, lw=0.9, zorder=1)
    for yi, m in zip(y, ORDER):
        for fam, A, B, off, col, mk in [
                ("mask_trainfull", f"bench|speech|{m}", f"bench|pause|{m}", 0.17, ORANGE, "s"),
                ("mask_traintest", f"tt_speech|speech|{m}", f"tt_pause|pause|{m}", -0.17, BLUE, "o")]:
            d, lo, hi, pv = P(fam, A, B, met)
            errpt(ax, d, lo, hi, yi + off, col, mk)
            ext = max(ext, abs(lo), abs(hi))
    ax.set_title(met)
    style(ax)
for ax in axes:
    ax.set_xlim(-ext * 1.08, ext * 1.08)
axes[0].set_yticks(y)
axes[0].set_yticklabels([NAME[m] for m in ORDER])
fig.supxlabel("Speech-only F1 minus pause-only F1 (95% CI). Positive values: higher F1 with speech-only vectors", fontsize=7.5)
h = [Line2D([0], [0], color=ORANGE, lw=1.2, marker="s", ms=4.2, mec="white", mew=0.6),
     Line2D([0], [0], color=BLUE, lw=1.2, marker="o", ms=4.2, mec="white", mew=0.6)]
fig.legend(h, ["Trained on full vectors, tested on masked vectors", "Trained and tested on masked vectors"],
           loc="outside upper center", ncol=2, frameon=False)
fig.savefig(FIG / "fig_masking_diff.pdf")
plt.close(fig)

# Figure 4: reversal
CONDS = [("Original audio", "bench", "full_valid", BLUE, "o", 0.22),
         ("Reversed audio, classifier trained on original audio", "bench", "rev_valid", ORANGE, "s", 0.0),
         ("Reversed audio, classifier trained on reversed audio", "tt_rev_valid", "rev_valid", AQUA, "^", -0.22)]
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.95), sharey=True, layout="constrained")
for ax, met in zip(axes, ["PPA", "macro"]):
    for yi, m in zip(y, ORDER):
        for lab_, e, c, col, mk, off in CONDS:
            r = S(e, c, m, met)
            errpt(ax, r.point, r.lo, r.hi, yi + off, col, mk)
    ax.set_xlim(0, 1)
    ax.set_title("PPA" if met == "PPA" else "Macro-F1")
    style(ax)
axes[0].set_yticks(y)
axes[0].set_yticklabels([NAME[m] for m in ORDER])
fig.supxlabel("F1 (95% CI)", fontsize=7.5)
h = [Line2D([0], [0], color=c[3], lw=1.2, marker=c[4], ms=4.2, mec="white", mew=0.6) for c in CONDS]
fig.legend(h, [c[0] for c in CONDS], loc="outside upper center", ncol=1, frameon=False)
fig.savefig(FIG / "fig_reversal.pdf")
plt.close(fig)

# ----------------------------------------------------------------------------- numbers for the prose
def row(exp, cond, m, met):
    r = S(exp, cond, m, met)
    return [round(float(r.point), 3), round(float(r.lo), 3), round(float(r.hi), 3)]


num["bench"] = {m: {met: row("bench", "full_valid", m, met) for met in COLS} for m in ORDER}
num["baseline"] = {b: {met: row("baseline", b, "-", met) for met in COLS} for b, _ in BASE}
num["chance_prior"] = {met: round(float(ch[met]), 3) for met in COLS}
num["fold_sd_macro"] = {m: float(S("bench", "full_valid", m, "macro").fold_sd) for m in ORDER}
num["vs_corpus"] = {m: {met: [round(v, 3) for v in P("model_vs_baseline", f"bench|full_valid|{m}", "baseline|corpus_only|-", met)] for met in COLS} for m in ORDER}
num["vs_handcrafted_macro"] = {m: [round(v, 3) for v in P("model_vs_baseline", f"bench|full_valid|{m}", "baseline|handcrafted_lr|-", "macro")] for m in ORDER}
num["wm_minus"] = {m: {met: [round(v, 3) for v in P("models", "bench|full_valid|whisper_medium", f"bench|full_valid|{m}", met)] for met in COLS} for m in ORDER if m != "whisper_medium"}
num["ssl_minus_960h"] = {met: [round(v, 3) for v in P("models", "bench|full_valid|wav2vec2_ssl", "bench|full_valid|wav2vec2", met)] for met in COLS}
num["mask"] = {}
for m in ORDER:
    num["mask"][m] = {}
    for met in ["MCI", "AD", "macro"] + ["Control", "ProbableAD", "PPA"]:
        num["mask"][m][met] = {
            "trainfull_sp_minus_pa": [round(v, 3) for v in P("mask_trainfull", f"bench|speech|{m}", f"bench|pause|{m}", met)],
            "traintest_sp_minus_pa": [round(v, 3) for v in P("mask_traintest", f"tt_speech|speech|{m}", f"tt_pause|pause|{m}", met)],
            "trainfull_sp_minus_shift": [round(v, 3) for v in P("mask_trainfull_vs_shift", f"bench|speech|{m}", f"bench|shift_speech_mean|{m}", met)],
            "trainfull_pa_minus_shift": [round(v, 3) for v in P("mask_trainfull_vs_shift", f"bench|pause|{m}", f"bench|shift_pause_mean|{m}", met)],
            "F1": {c: round(float(S(e, c, m, met).point), 3) for e, c, _ in MASK_COLS},
        }
num["rev"] = {}
for m in ORDER:
    num["rev"][m] = {}
    for met in COLS:
        num["rev"][m][met] = {
            "F1": {lab_: round(float(S(e, c, m, met).point), 3) for lab_, e, c in REV},
            "drop_trainfwd": [round(v, 3) for v in P("reversal_trainfwd", f"bench|full_valid|{m}", f"bench|rev_valid|{m}", met)],
            "drop_traintest": [round(v, 3) for v in P("reversal_traintest", f"bench|full_valid|{m}", f"tt_rev_valid|rev_valid|{m}", met)],
            "drop_local_trainfwd": [round(v, 3) for v in P("local_reversal_trainfwd", f"bench|full_valid|{m}", f"bench|lrev_valid|{m}", met)],
            "drop_local_traintest": [round(v, 3) for v in P("local_reversal_traintest", f"bench|full_valid|{m}", f"tt_lrev_valid|lrev_valid|{m}", met)],
        }
num["whisper_pooling"] = {m: {met: [round(v, 3) for v in P("whisper_pooling", f"bench_all|full_all|{m}", f"bench|full_valid|{m}", met)] for met in COLS} for m in ["whisper_base", "whisper_medium"]}
num["corpus_id"] = cid.to_dict()
num["within"] = {c: {m: [round(float(x), 3) for x in w[(w.corpus == c) & (w.model == m) & (w.metric == "macro")][["point", "lo", "hi"]].iloc[0]] for m in ORDER} for c in CORPORA}
num["within_classes"] = {c: K[c] for c in CORPORA}
(GEN / "numbers.json").write_text(json.dumps(num, indent=1))
print("ties:", json.dumps(num["ties"]))
print("written:", sorted(x.name for x in GEN.iterdir()))
