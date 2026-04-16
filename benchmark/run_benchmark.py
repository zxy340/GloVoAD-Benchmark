"""
Step 2: Five-class ADRD subtype benchmark with speaker-independent splits.

Uses StratifiedGroupKFold to ensure speaker independence (same speaker's
multiple recordings never appear in both train and test). Speaker IDs are
derived from Pitt corpus filenames; other corpora use filename as speaker ID.

Includes both pretrained model benchmark and handcrafted feature baseline.
Produces Table 3 in the paper.

Usage:
    python run_benchmark.py \
        --manifest manifest.csv \
        --feat_dir features/ \
        --device cpu

    # With audio root override (for local reproduction):
    python run_benchmark.py \
        --manifest manifest.csv \
        --feat_dir features/ \
        --data_root "/path/to/Original Audios" \
        --device cpu
"""
import os
import csv
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
import librosa
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

DIAGNOSIS = {"Control": 0, "MCI": 1, "AD": 2, "ProbableAD": 3, "PPA": 4}
NAMES = {0: "Control", 1: "MCI", 2: "AD", 3: "ProbableAD", 4: "PPA"}
NUM_CLASSES = 5
SEEDS = [42, 123, 456]
N_FOLDS = 5

MODELS = [
    "wav2vec2", "hubert", "wavlm", "whisper_base",
    "xlsr", "whisper_medium", "unispeech_sat",
]


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


def remap_path(row, data_root):
    """Remap audio path if data_root is provided (for local reproduction)."""
    if data_root:
        return os.path.join(data_root, row["diagnosis"], row["corpus"], row["filename"])
    return row["path"]


def get_speaker_id(row):
    """Extract speaker ID for grouping. Pitt corpus has multiple recordings
    per speaker (e.g., '001-0.mp3', '001-1.mp3' → speaker '001')."""
    if row["corpus"] == "Pitt":
        match = re.match(r"(\d+)-", row["filename"])
        if match:
            return f"Pitt_{row['diagnosis']}_{match.group(1)}"
    return f"{row['corpus']}_{row['diagnosis']}_{row['filename']}"


def load_features(model_name, rows, feat_dir):
    """Load precomputed features and labels, return (X, y, groups)."""
    fdir = Path(feat_dir) / model_name
    X, labels, groups = [], [], []
    for row in rows:
        fname = f"{row['corpus']}_{row['diagnosis']}_{row['filename']}.npy"
        fp = fdir / fname
        if not fp.exists():
            continue
        X.append(np.load(fp).flatten())
        labels.append(DIAGNOSIS[row["diagnosis"]])
        groups.append(get_speaker_id(row))
    if not X:
        return None, None, None
    return np.stack(X), np.array(labels), np.array(groups)


def evaluate_model(X, y, groups, device):
    """5-fold StratifiedGroupKFold × 3 seeds → macro F1 + per-class F1."""
    fold_f1s, fold_pers = [], []
    all_preds, all_trues = [], []

    for seed in SEEDS:
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr, te in sgkf.split(X, y, groups):
            sc = StandardScaler()
            Xtr = torch.FloatTensor(sc.fit_transform(X[tr])).to(device)
            Xte = torch.FloatTensor(sc.transform(X[te])).to(device)
            ytr = torch.LongTensor(y[tr]).to(device)

            cc = np.bincount(y[tr], minlength=NUM_CLASSES).astype(float)
            cw = 1.0 / (cc + 1e-6)
            cw = cw / cw.sum() * NUM_CLASSES

            mdl = MLP(X.shape[1], NUM_CLASSES).to(device)
            opt = torch.optim.Adam(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
            crit = nn.CrossEntropyLoss(weight=torch.FloatTensor(cw).to(device))
            ds = TensorDataset(Xtr, ytr)
            loader = DataLoader(ds, batch_size=64, shuffle=True)

            mdl.train()
            for _ in range(80):
                for xb, yb in loader:
                    loss = crit(mdl(xb), yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            mdl.eval()
            with torch.no_grad():
                preds = mdl(Xte).argmax(1).cpu().numpy()

            fold_f1s.append(f1_score(y[te], preds, average="macro", zero_division=0))
            fold_pers.append(f1_score(y[te], preds, average=None, zero_division=0))
            all_preds.extend(preds.tolist())
            all_trues.extend(y[te].tolist())

    macro_f1 = np.mean(fold_f1s)
    macro_std = np.std(fold_f1s)
    per_class = np.mean(fold_pers, axis=0)
    cm = confusion_matrix(all_trues, all_preds, labels=list(range(NUM_CLASSES)))
    return macro_f1, macro_std, per_class, cm


TARGET_SR = 16000


def extract_librosa_features(path):
    """Extract 37 handcrafted acoustic features using librosa."""
    y, sr = librosa.load(path, sr=TARGET_SR, duration=30)
    if len(y) < TARGET_SR:
        return None

    features = []

    # MFCCs (13) + delta MFCCs (13)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(mfcc.mean(axis=1))
    delta_mfcc = librosa.feature.delta(mfcc)
    features.extend(delta_mfcc.mean(axis=1))

    # Spectral features (5)
    features.append(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    features.append(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
    features.append(librosa.feature.spectral_contrast(y=y, sr=sr).mean())
    features.append(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
    features.append(librosa.feature.spectral_flatness(y=y).mean())

    # Temporal features (2)
    features.append(librosa.feature.zero_crossing_rate(y).mean())
    features.append(librosa.feature.rms(y=y).mean())

    # Chroma (1)
    features.append(librosa.feature.chroma_stft(y=y, sr=sr).mean())

    # Tempo (1)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(tempo) if np.isscalar(tempo) else float(tempo[0]))

    # Pitch (2)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[magnitudes > np.median(magnitudes)]
    pitch_vals = pitch_vals[pitch_vals > 0]
    features.append(pitch_vals.mean() if len(pitch_vals) > 0 else 0.0)
    features.append(pitch_vals.std() if len(pitch_vals) > 0 else 0.0)

    return np.array(features, dtype=np.float32)


def run_handcrafted_baseline(rows, groups_all, data_root=None):
    """Extract handcrafted features and evaluate with SVM."""
    print("\nExtracting handcrafted (librosa) features...")
    X_list, y_list, groups_list = [], [], []
    for i, row in enumerate(rows):
        try:
            audio_path = remap_path(row, data_root)
            feat = extract_librosa_features(audio_path)
            if feat is None:
                continue
            X_list.append(feat)
            y_list.append(DIAGNOSIS[row["diagnosis"]])
            groups_list.append(get_speaker_id(row))
        except Exception as e:
            if i < 3:  # print first few errors for debugging
                print(f"  [WARN] {row['filename']}: {e}")
            continue
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}, extracted={len(X_list)}")

    if not X_list:
        print("  ERROR: No features extracted. Install ffmpeg: brew install ffmpeg")
        return 0.0, 0.0, np.zeros(NUM_CLASSES)

    X = np.stack(X_list)
    y = np.array(y_list)
    groups = np.array(groups_list)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Extracted: {X.shape[0]} samples, {X.shape[1]} features")

    # Evaluate with SVM + StratifiedGroupKFold
    fold_f1s, fold_pers = [], []
    for seed in SEEDS:
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr, te in sgkf.split(X, y, groups):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X[tr])
            X_te = sc.transform(X[te])
            lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
            lr.fit(X_tr, y[tr])
            preds = lr.predict(X_te)
            fold_f1s.append(f1_score(y[te], preds, average="macro", zero_division=0))
            fold_pers.append(f1_score(y[te], preds, average=None, zero_division=0))

    macro_f1 = np.mean(fold_f1s)
    macro_std = np.std(fold_f1s)
    per_class = np.mean(fold_pers, axis=0)
    return macro_f1, macro_std, per_class


def main():
    parser = argparse.ArgumentParser(description="5-class ADRD benchmark")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--feat_dir", type=str, default="features")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override audio root dir for handcrafted baseline "
                             "(e.g., /path/to/Original Audios)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))

    # Count unique speakers
    all_speakers = set(get_speaker_id(r) for r in rows)
    print(f"Total files: {len(rows)}, Unique speakers: {len(all_speakers)}")

    print(f"\n{'='*100}")
    print("BENCHMARK WITH SPEAKER-INDEPENDENT SPLIT (StratifiedGroupKFold)")
    print(f"{'='*100}")
    print(f"{'Model':24s} {'F1 Macro':>12s} "
          + " ".join(f"{NAMES[c]:>10s}" for c in range(5)))
    print("-" * 100)

    best_cm = None
    for mname in MODELS:
        X, y, groups = load_features(mname, rows, args.feat_dir)
        if X is None:
            print(f"{mname:24s} [SKIP - no features]")
            continue

        f1, std, per, cm = evaluate_model(X, y, groups, args.device)
        print(f"{mname:24s} {f1:.4f}\u00b1{std:.4f} "
              + " ".join(f"{per[c]:10.4f}" for c in range(5)))

        if mname == "whisper_medium":
            best_cm = cm

    # Handcrafted baseline
    print("-" * 100)
    hc_f1, hc_std, hc_per = run_handcrafted_baseline(rows, None, args.data_root)
    print(f"{'Handcrafted+LR':24s} {hc_f1:.4f}\u00b1{hc_std:.4f} "
          + " ".join(f"{hc_per[c]:10.4f}" for c in range(5)))

    # Print confusion matrix after the table
    if best_cm is not None:
        cm_norm = best_cm.astype(float) / best_cm.sum(axis=1, keepdims=True)
        print(f"\nConfusion Matrix (Whisper-medium, normalized):")
        print(f"  {'':12s} " + " ".join(f"{NAMES[c]:>10s}" for c in range(5)))
        for i in range(5):
            print(f"  {NAMES[i]:12s} " + " ".join(f"{cm_norm[i,j]:10.3f}" for j in range(5)))


if __name__ == "__main__":
    main()
