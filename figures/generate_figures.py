"""
Generate all 4 publication-quality figures for the paper.

Figures:
  1. fig_benchmark_bars.pdf  — Per-class F1 grouped bar chart (Figure 1)
  2. fig_region_masking.pdf  — Region masking heatmap (Figure 2)
  3. fig_ppa_drop.pdf        — PPA detection drop bar chart (Figure 3)
  4. fig_causal_chain.pdf    — Mechanistic evidence chain diagram (Figure 4)

Usage:
    python generate_figures.py --output_dir figures/
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ============================================================
# Data from paper (Table 3, Figure 2, Table 5)
# Update these values if you re-run experiments
# ============================================================

MODELS = ["wav2vec2", "hubert", "wavlm", "whisper_base", "xlsr", "whisper_medium", "unispeech_sat"]
MODEL_LABELS = ["Wav2Vec2", "HuBERT", "WavLM", "Whisper-base", "XLS-R", "Whisper-med", "UniSpeech-SAT"]
CLASSES = ["Control", "MCI", "AD", "ProbableAD", "PPA"]

# Table 3: Per-class F1 from GroupKFold benchmark
BENCHMARK = {
    "wav2vec2":       [0.630, 0.406, 0.286, 0.441, 0.370],
    "hubert":         [0.685, 0.458, 0.453, 0.512, 0.681],
    "wavlm":         [0.676, 0.500, 0.409, 0.520, 0.651],
    "whisper_base":   [0.714, 0.488, 0.396, 0.565, 0.774],
    "xlsr":           [0.631, 0.443, 0.445, 0.469, 0.582],
    "whisper_medium": [0.712, 0.484, 0.425, 0.559, 0.801],
    "unispeech_sat":  [0.674, 0.475, 0.421, 0.529, 0.644],
}

# Figure 2: Region masking — AD/MCI dependency
# "Pause" = model depends on pause, "Speech" = model depends on speech
REGION_MASKING = {
    "wav2vec2":       {"AD": "Speech", "MCI": "Speech"},
    "hubert":         {"AD": "Speech", "MCI": "Speech"},
    "wavlm":         {"AD": "Pause",  "MCI": "Pause"},
    "whisper_base":   {"AD": "Speech", "MCI": "Pause"},
    "xlsr":           {"AD": "Speech", "MCI": "Speech"},
    "whisper_medium": {"AD": "Pause",  "MCI": "Pause"},
    "unispeech_sat":  {"AD": "Pause",  "MCI": "Pause"},
}

# Table 5: Linguistic perturbation — PPA drop
PPA_DROP = {
    "wav2vec2":       {"orig": 0.372, "rev": 0.082, "drop": 0.290},
    "hubert":         {"orig": 0.682, "rev": 0.295, "drop": 0.387},
    "wavlm":         {"orig": 0.673, "rev": 0.272, "drop": 0.401},
    "whisper_base":   {"orig": 0.760, "rev": 0.182, "drop": 0.577},
    "xlsr":           {"orig": 0.492, "rev": 0.127, "drop": 0.365},
    "whisper_medium": {"orig": 0.844, "rev": 0.064, "drop": 0.780},
    "unispeech_sat":  {"orig": 0.694, "rev": 0.127, "drop": 0.567},
}


# ============================================================
# Figure 1: Benchmark bar chart
# ============================================================
def fig_benchmark_bars(output_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(MODELS))
    width = 0.15
    colors = ["#2C73D2", "#FF9671", "#FFC75F", "#845EC2", "#D65DB1"]

    for ci, cls in enumerate(CLASSES):
        vals = [BENCHMARK[m][ci] for m in MODELS]
        ax.bar(x + ci * width, vals, width, label=cls, color=colors[ci], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(MODEL_LABELS, fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper right", fontsize=8, ncol=5)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(output_dir, "fig_benchmark_bars.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================
# Figure 2: Region masking heatmap (2×7 grid)
# ============================================================
def fig_region_masking(output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
    subtypes = ["AD", "MCI"]
    titles = ["AD Dependence", "MCI Dependence"]

    for idx, (subtype, title) in enumerate(zip(subtypes, titles)):
        ax = axes[idx]
        vals = []
        colors = []
        for m in MODELS:
            dep = REGION_MASKING[m][subtype]
            vals.append(dep)
            colors.append("#E74C3C" if dep == "Pause" else "#3498DB")

        y_pos = np.arange(len(MODELS))
        bars = ax.barh(y_pos, [1] * len(MODELS), color=colors, edgecolor="white", height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(MODEL_LABELS, fontsize=9)
        ax.set_xticks([])
        ax.set_title(title, fontsize=11, pad=8)

        for i, (bar, dep) in enumerate(zip(bars, vals)):
            ax.text(0.5, i, dep, ha="center", va="center", fontsize=9, fontweight="bold", color="white")

        ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#E74C3C", label="Pause-dependent"),
                      Patch(facecolor="#3498DB", label="Speech-dependent")]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, frameon=False, fontsize=9,
              bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    path = os.path.join(output_dir, "fig_region_masking.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================
# Figure 3: PPA detection drop bar chart
# ============================================================
def fig_ppa_drop(output_dir):
    fig, ax = plt.subplots(figsize=(8, 4))

    # Sort by drop magnitude
    sorted_models = sorted(MODELS, key=lambda m: PPA_DROP[m]["drop"], reverse=True)
    sorted_labels = [MODEL_LABELS[MODELS.index(m)] for m in sorted_models]

    drops = [PPA_DROP[m]["drop"] for m in sorted_models]
    colors = ["#E74C3C" if d > 0.5 else "#FF9671" if d > 0.4 else "#FFC75F" for d in drops]

    x = np.arange(len(sorted_models))
    bars = ax.bar(x, drops, color=colors, edgecolor="white", linewidth=0.5)

    for i, (bar, d) in enumerate(zip(bars, drops)):
        ax.text(i, d + 0.01, f"{d:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("PPA F1 Drop (Original − Reversed)")
    ax.set_ylim(0, 0.95)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(output_dir, "fig_ppa_drop.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================
# Figure 4: Causal chain diagram
# ============================================================
def fig_causal_chain(output_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def draw_box(x, y, w, h, text, color, fontsize=8):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor="black", linewidth=1)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
               fontsize=fontsize, wrap=True)

    # Column headers
    headers = ["Training\nObjective", "Feature\nBias", "Subtype\nSensitivity", "Clinical\nPathology"]
    hx = [0.5, 3.5, 6.5, 9.5]
    for x, h in zip(hx, headers):
        ax.text(x + 1, 4.6, h, ha="center", va="center", fontsize=10, fontweight="bold")

    # Path 1: Denoising → Temporal → AD/MCI → Temporal disruption
    y1 = 3.0
    draw_box(0.3, y1, 2.0, 0.8, "Denoising\n(WavLM)", "#AED6F1")
    draw_box(3.3, y1, 2.0, 0.8, "Temporal\nstructure", "#ABEBC6")
    draw_box(6.3, y1, 2.0, 0.8, "AD / MCI\nsensitivity", "#F9E79F")
    draw_box(9.3, y1, 2.0, 0.8, "Hesitation &\npause patterns", "#F5CBA7")

    # Path 2: ASR → Linguistic → PPA → Language breakdown
    y2 = 1.2
    draw_box(0.3, y2, 2.0, 0.8, "ASR\n(Whisper)", "#D2B4DE")
    draw_box(3.3, y2, 2.0, 0.8, "Linguistic\ncontent", "#F1948A")
    draw_box(6.3, y2, 2.0, 0.8, "PPA\nsensitivity", "#F9E79F")
    draw_box(9.3, y2, 2.0, 0.8, "Word-finding &\nphonological errors", "#F5CBA7")

    # Arrows
    arrow_style = dict(arrowstyle="->", color="black", lw=1.5)
    for y in [y1 + 0.4, y2 + 0.4]:
        for x_start, x_end in [(2.3, 3.3), (5.3, 6.3), (8.3, 9.3)]:
            ax.annotate("", xy=(x_end, y), xytext=(x_start, y), arrowprops=arrow_style)

    # Path labels
    ax.text(-0.2, y1 + 0.4, "Path 1", fontsize=9, fontweight="bold", va="center", color="#2980B9")
    ax.text(-0.2, y2 + 0.4, "Path 2", fontsize=9, fontweight="bold", va="center", color="#8E44AD")

    path = os.path.join(output_dir, "fig_causal_chain.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--output_dir", type=str, default="figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Generating figures...")
    fig_benchmark_bars(args.output_dir)
    fig_region_masking(args.output_dir)
    fig_ppa_drop(args.output_dir)
    fig_causal_chain(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
