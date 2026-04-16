"""
Step 3: Region masking experiment — speech vs. pause dependency analysis.

Uses Silero VAD to segment audio into speech/pause frames, then tests each
model with speech-only and pause-only features to determine which temporal
region is more important for each ADRD subtype.

Produces Figure 2 in the paper.

Usage:
    python run_region_masking.py \
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

# Monkey-patch torch.load for Silero VAD and older model checkpoints
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(
    *args, **{k: v for k, v in kwargs.items() if k != "weights_only"}, weights_only=False
)

import torchaudio
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
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


def run_region_masking(mname, config, rows, device, data_root=None):
    """Extract full, speech-only, and pause-only features; evaluate each."""
    print(f"\n{'='*60}")
    print(f"Region Masking: {mname}")
    print(f"{'='*60}")

    model = config["cls"].from_pretrained(config["id"]).to(device).eval()
    proc = config["proc_cls"].from_pretrained(config["id"])
    fdim = config["dim"]

    # Load Silero VAD
    vad_model, vad_utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", force_reload=False, trust_repo=True
    )
    get_speech_ts = vad_utils[0]

    def get_frames(wav):
        with torch.no_grad():
            if config["whisper_style"]:
                inp = proc(wav.numpy(), sampling_rate=TARGET_SR, return_tensors="pt")
                out = model.encoder(inp.input_features.to(device))
            else:
                inp = proc(wav.numpy(), sampling_rate=TARGET_SR,
                          return_tensors="pt", padding=True)
                out = model(inp.input_values.to(device))
            return out.last_hidden_state.squeeze(0).cpu().numpy()

    X_full, X_speech, X_pause, y_all, groups = [], [], [], [], []

    for i, row in enumerate(rows):
        try:
            audio_path = remap_path(row, data_root)
            wav = load_audio(audio_path)
            if wav.shape[0] < TARGET_SR:
                continue

            ts = get_speech_ts(wav, vad_model, sampling_rate=TARGET_SR,
                              threshold=0.5, min_speech_duration_ms=100,
                              min_silence_duration_ms=100)

            frames = get_frames(wav)
            T = frames.shape[0]

            mask = np.zeros(T, dtype=bool)
            for t in ts:
                s = int(t["start"] / wav.shape[0] * T)
                e = int(t["end"] / wav.shape[0] * T)
                mask[s:min(e, T)] = True

            sp = frames[mask]
            pp = frames[~mask]
            X_full.append(frames.mean(0))
            X_speech.append(sp.mean(0) if len(sp) > 0 else np.zeros(fdim))
            X_pause.append(pp.mean(0) if len(pp) > 0 else np.zeros(fdim))
            y_all.append(DIAGNOSIS[row["diagnosis"]])
            groups.append(get_speaker_id(row))
        except:
            continue
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(rows)}")

    X_full = np.stack(X_full)
    X_speech = np.stack(X_speech)
    X_pause = np.stack(X_pause)
    y = np.array(y_all)
    groups = np.array(groups)

    del model
    torch.cuda.empty_cache()

    # Evaluate: train on full, test on full/speech-only/pause-only
    per_full, per_speech, per_pause = [], [], []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr, te in skf.split(X_full, y):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_full[tr])

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
                pf = f1_score(y[te], mlp(torch.FloatTensor(sc.transform(X_full[te])).to(device)).argmax(1).cpu().numpy(), average=None, zero_division=0)
                ps = f1_score(y[te], mlp(torch.FloatTensor(sc.transform(X_speech[te])).to(device)).argmax(1).cpu().numpy(), average=None, zero_division=0)
                pp = f1_score(y[te], mlp(torch.FloatTensor(sc.transform(X_pause[te])).to(device)).argmax(1).cpu().numpy(), average=None, zero_division=0)
            per_full.append(pf)
            per_speech.append(ps)
            per_pause.append(pp)

    avg_full = np.mean(per_full, axis=0)
    avg_speech = np.mean(per_speech, axis=0)
    avg_pause = np.mean(per_pause, axis=0)

    # Compute dependency for each subtype
    results = {}
    print(f"\n  {'Subtype':12s} {'Full':>8s} {'Speech':>8s} {'Pause':>8s} {'Drop(-pause)':>13s} {'Drop(-speech)':>14s} {'Depends on':>12s}")
    print(f"  {'-'*85}")
    for c in range(NUM_CLASSES):
        drop_no_pause = avg_full[c] - avg_speech[c]  # drop when pause removed
        drop_no_speech = avg_full[c] - avg_pause[c]   # drop when speech removed
        depends = "Pause" if drop_no_pause > drop_no_speech else "Speech"
        results[NAMES[c]] = depends
        print(f"  {NAMES[c]:12s} {avg_full[c]:8.4f} {avg_speech[c]:8.4f} {avg_pause[c]:8.4f} {drop_no_pause:+13.4f} {drop_no_speech:+14.4f} {depends:>12s}")

    ad_dep = results["AD"]
    mci_dep = results["MCI"]
    print(f"\n  >>> AD Dependence: {ad_dep}    MCI Dependence: {mci_dep}")

    return {"ad_dep": ad_dep, "mci_dep": mci_dep}


def main():
    parser = argparse.ArgumentParser(description="Region masking experiment")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--model", type=str, default="all",
                        choices=list(MODEL_CONFIGS.keys()) + ["all"])
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override audio root dir (e.g., /local/path/to/Original Audios)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))

    MODEL_LABELS = {
        "wav2vec2": "wav2vec2", "hubert": "HuBERT", "wavlm": "WavLM",
        "whisper_base": "Whisper-Base", "xlsr": "XLS-R",
        "whisper_medium": "Whisper-Medium", "unispeech_sat": "UniSpeech-SAT",
    }

    models = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    all_results = {}
    for mname in models:
        res = run_region_masking(mname, MODEL_CONFIGS[mname], rows, args.device, args.data_root)
        all_results[mname] = res

    # Summary table matching Figure 2
    print(f"\n{'='*60}")
    print("SUMMARY: Region Masking Results (Figure 2)")
    print(f"{'='*60}")
    print(f"{'Model':20s} {'AD Dependence':>16s} {'MCI Dependence':>16s}")
    print(f"{'-'*54}")
    for mname in models:
        if mname in all_results:
            label = MODEL_LABELS.get(mname, mname)
            ad = all_results[mname]["ad_dep"]
            mci = all_results[mname]["mci_dep"]
            print(f"{label:20s} {ad:>16s} {mci:>16s}")


if __name__ == "__main__":
    main()
