"""
Step 4: Linguistic perturbation experiment — audio reversal.

Reverses audio to destroy linguistic content while preserving acoustic
properties, then measures PPA detection drop. ASR-trained models (Whisper)
show the largest drops, confirming their reliance on linguistic content.

Produces Table 5 and Figure 3 in the paper.

Usage:
    python run_linguistic_perturbation.py \
        --manifest manifest.csv \
        --device cuda:0
"""
import csv
import re
import argparse
import os
import warnings

# Suppress noisy MP3 decoding warnings from librosa/audioread
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_ALLOW_UNSAFE_DESERIALIZATION"] = "1"

import numpy as np
import torch
import torch.nn as nn

# Monkey-patch torch.load for older model checkpoints
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(
    *args, **{k: v for k, v in kwargs.items() if k != "weights_only"}, weights_only=False
)

import torchaudio
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    Wav2Vec2Model, Wav2Vec2FeatureExtractor,
    HubertModel, WavLMModel,
    WhisperModel, WhisperProcessor,
    UniSpeechSatModel,
)

TARGET_SR = 16000
MAX_DUR = 30
DIAGNOSIS = {"Control": 0, "MCI": 1, "AD": 2, "ProbableAD": 3, "PPA": 4}
NAMES = {0: "Control", 1: "MCI", 2: "AD", 3: "ProbableAD", 4: "PPA"}
NUM_CLASSES = 5
SEEDS = [42, 123, 456]
N_FOLDS = 5

MODEL_CONFIGS = {
    "wav2vec2": {
        "id": "facebook/wav2vec2-base-960h",
        "cls": Wav2Vec2Model,
        "proc_cls": Wav2Vec2FeatureExtractor,
        "whisper_style": False,
        "dim": 768,
    },
    "hubert": {
        "id": "facebook/hubert-base-ls960",
        "cls": HubertModel,
        "proc_cls": Wav2Vec2FeatureExtractor,
        "whisper_style": False,
        "dim": 768,
    },
    "wavlm": {
        "id": "microsoft/wavlm-base-plus",
        "cls": WavLMModel,
        "proc_cls": Wav2Vec2FeatureExtractor,
        "whisper_style": False,
        "dim": 768,
    },
    "whisper_base": {
        "id": "openai/whisper-base",
        "cls": WhisperModel,
        "proc_cls": WhisperProcessor,
        "whisper_style": True,
        "dim": 512,
    },
    "xlsr": {
        "id": "facebook/wav2vec2-xls-r-300m",
        "cls": Wav2Vec2Model,
        "proc_cls": Wav2Vec2FeatureExtractor,
        "whisper_style": False,
        "dim": 1024,
    },
    "whisper_medium": {
        "id": "openai/whisper-medium",
        "cls": WhisperModel,
        "proc_cls": WhisperProcessor,
        "whisper_style": True,
        "dim": 1024,
    },
    "unispeech_sat": {
        "id": "microsoft/unispeech-sat-base-plus",
        "cls": UniSpeechSatModel,
        "proc_cls": Wav2Vec2FeatureExtractor,
        "whisper_style": False,
        "dim": 768,
    },
}


class MLP(nn.Module):
    def __init__(self, d, nc, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, nc),
        )

    def forward(self, x):
        return self.net(x)


def load_audio(path):
    w, sr = torchaudio.load(path)
    if w.shape[0] > 1:
        w = w.mean(0, keepdim=True)
    if sr != TARGET_SR:
        w = torchaudio.functional.resample(w, sr, TARGET_SR)
    w = w.squeeze(0)
    if w.shape[0] > MAX_DUR * TARGET_SR:
        w = w[:MAX_DUR * TARGET_SR]
    return w


def get_speaker_id(row):
    if row["corpus"] == "Pitt":
        match = re.match(r"(\d+)-", row["filename"])
        if match:
            return f"Pitt_{row['diagnosis']}_{match.group(1)}"
    return f"{row['corpus']}_{row['diagnosis']}_{row['filename']}"


def remap_path(row, data_root):
    """Remap audio path if data_root is provided (for local reproduction)."""
    if data_root:
        return os.path.join(data_root, row["diagnosis"], row["corpus"], row["filename"])
    return row["path"]


def run_perturbation(mname, config, rows, device, data_root=None):
    """Extract original and reversed features; compare PPA detection."""
    print(f"\n{'='*60}")
    print(f"Linguistic Perturbation: {mname}")
    print(f"{'='*60}")

    model = config["cls"].from_pretrained(config["id"]).to(device).eval()
    proc = config["proc_cls"].from_pretrained(config["id"])
    fdim = config["dim"]

    def get_emb(wav):
        with torch.no_grad():
            if config["whisper_style"]:
                inp = proc(wav.numpy(), sampling_rate=TARGET_SR, return_tensors="pt")
                out = model.encoder(inp.input_features.to(device))
            else:
                inp = proc(wav.numpy(), sampling_rate=TARGET_SR,
                          return_tensors="pt", padding=True)
                out = model(inp.input_values.to(device))
            return out.last_hidden_state.squeeze(0).mean(0).cpu().numpy()

    X_orig, X_rev, y_all, groups = [], [], [], []

    for i, row in enumerate(rows):
        try:
            audio_path = remap_path(row, data_root)
            wav = load_audio(audio_path)
            if wav.shape[0] < TARGET_SR:
                continue
            X_orig.append(get_emb(wav))
            X_rev.append(get_emb(wav.flip(0)))  # reverse audio
            y_all.append(DIAGNOSIS[row["diagnosis"]])
            groups.append(get_speaker_id(row))
        except:
            continue
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(rows)}")

    X_orig = np.stack(X_orig)
    X_rev = np.stack(X_rev)
    y = np.array(y_all)
    groups = np.array(groups)

    del model
    torch.cuda.empty_cache()

    # Evaluate original and reversed
    per_orig, per_rev = [], []
    for seed in SEEDS:
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr, te in sgkf.split(X_orig, y, groups):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_orig[tr])

            cc = np.bincount(y[tr], minlength=NUM_CLASSES).astype(float)
            cw = 1.0 / (cc + 1e-6)
            cw = cw / cw.sum() * NUM_CLASSES

            mlp = MLP(fdim, NUM_CLASSES).to(device)
            opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
            crit = nn.CrossEntropyLoss(weight=torch.FloatTensor(cw).to(device))
            ds = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y[tr]))
            loader = DataLoader(ds, batch_size=64, shuffle=True)

            mlp.train()
            for _ in range(80):
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    loss = crit(mlp(xb), yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            mlp.eval()
            with torch.no_grad():
                po = f1_score(y[te], mlp(torch.FloatTensor(sc.transform(X_orig[te])).to(device)).argmax(1).cpu().numpy(), average=None, zero_division=0)
                pr = f1_score(y[te], mlp(torch.FloatTensor(sc.transform(X_rev[te])).to(device)).argmax(1).cpu().numpy(), average=None, zero_division=0)
            per_orig.append(po)
            per_rev.append(pr)

    avg_orig = np.mean(per_orig, axis=0)
    avg_rev = np.mean(per_rev, axis=0)

    # Focus on PPA
    ppa_idx = 4
    ppa_orig = avg_orig[ppa_idx]
    ppa_rev = avg_rev[ppa_idx]
    ppa_drop = ppa_orig - ppa_rev

    print(f"\n  {'Subtype':12s} {'Original':>10s} {'Reversed':>10s} {'Drop':>10s}")
    print(f"  {'-'*44}")
    for c in range(NUM_CLASSES):
        drop = avg_orig[c] - avg_rev[c]
        print(f"  {NAMES[c]:12s} {avg_orig[c]:10.4f} {avg_rev[c]:10.4f} {drop:+10.4f}")

    print(f"\n  >>> PPA drop: {ppa_drop:+.4f} (Original={ppa_orig:.4f}, Reversed={ppa_rev:.4f})")

    return {"ppa_orig": ppa_orig, "ppa_rev": ppa_rev, "ppa_drop": ppa_drop}


def main():
    parser = argparse.ArgumentParser(description="Linguistic perturbation experiment")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--model", type=str, default="all",
                        choices=list(MODEL_CONFIGS.keys()) + ["all"])
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override audio root dir (e.g., /local/path/to/Original Audios)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))

    models = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    all_results = {}
    for mname in models:
        res = run_perturbation(mname, MODEL_CONFIGS[mname], rows, args.device, args.data_root)
        all_results[mname] = res

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: PPA detection drop after audio reversal")
    print(f"{'='*60}")
    print(f"{'Model':20s} {'PPA Orig':>10s} {'PPA Rev':>10s} {'Drop':>10s}")
    print("-" * 52)
    for m, r in sorted(all_results.items(), key=lambda x: -x[1]["ppa_drop"]):
        print(f"{m:20s} {r['ppa_orig']:10.4f} {r['ppa_rev']:10.4f} {r['ppa_drop']:+10.4f}")


if __name__ == "__main__":
    main()
