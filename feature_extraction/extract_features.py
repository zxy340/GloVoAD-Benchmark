"""
Step 1: Extract mean-pooled features from 7 pretrained speech models.

Models:
  - wav2vec2     (facebook/wav2vec2-base-960h)       — Contrastive SSL, dim=768
  - hubert       (facebook/hubert-base-ls960)        — Cluster Prediction, dim=768
  - wavlm        (microsoft/wavlm-base-plus)         — Cluster+Denoising, dim=768
  - whisper_base (openai/whisper-base)                — ASR Supervised, dim=512
  - xlsr         (facebook/wav2vec2-xls-r-300m)       — Multilingual SSL, dim=1024
  - whisper_medium (openai/whisper-medium)            — ASR Supervised (Large), dim=1024
  - unispeech_sat (microsoft/unispeech-sat-base-plus) — Speaker-Aware SSL, dim=768

Usage:
    python extract_features.py --manifest manifest.csv --output_dir features/ --device cuda:0
    python extract_features.py --manifest manifest.csv --output_dir features/ --model wav2vec2
"""
import csv
import argparse
import numpy as np
import torch
import librosa
from pathlib import Path
from transformers import (
    Wav2Vec2Model, Wav2Vec2FeatureExtractor,
    HubertModel, WavLMModel,
    WhisperModel, WhisperProcessor,
    UniSpeechSatModel,
)

TARGET_SR = 16000
MAX_DURATION = 30  # seconds

MODEL_CONFIGS = {
    "wav2vec2": {
        "id": "facebook/wav2vec2-base-960h",
        "cls": Wav2Vec2Model,
        "proc_cls": Wav2Vec2FeatureExtractor,
        "whisper_style": False,
        "dim": 768,
        "category": "Contrastive SSL",
    },
    "hubert": {
        "id": "facebook/hubert-base-ls960",
        "cls": HubertModel,
        "proc_cls": Wav2Vec2FeatureExtractor,
        "whisper_style": False,
        "dim": 768,
        "category": "Cluster Prediction",
    },
    "wavlm": {
        "id": "microsoft/wavlm-base-plus",
        "cls": WavLMModel,
        "proc_cls": Wav2Vec2FeatureExtractor,
        "whisper_style": False,
        "dim": 768,
        "category": "Cluster+Denoising",
    },
    "whisper_base": {
        "id": "openai/whisper-base",
        "cls": WhisperModel,
        "proc_cls": WhisperProcessor,
        "whisper_style": True,
        "dim": 512,
        "category": "ASR Supervised",
    },
    "xlsr": {
        "id": "facebook/wav2vec2-xls-r-300m",
        "cls": Wav2Vec2Model,
        "proc_cls": Wav2Vec2FeatureExtractor,
        "whisper_style": False,
        "dim": 1024,
        "category": "Multilingual SSL",
    },
    "whisper_medium": {
        "id": "openai/whisper-medium",
        "cls": WhisperModel,
        "proc_cls": WhisperProcessor,
        "whisper_style": True,
        "dim": 1024,
        "category": "ASR Supervised (Large)",
    },
    "unispeech_sat": {
        "id": "microsoft/unispeech-sat-base-plus",
        "cls": UniSpeechSatModel,
        "proc_cls": Wav2Vec2FeatureExtractor,
        "whisper_style": False,
        "dim": 768,
        "category": "Speaker-Aware SSL",
    },
}


def load_audio(path):
    """Load and preprocess audio: mono, resample to 16kHz, truncate to 30s."""
    y, sr = librosa.load(path, sr=TARGET_SR, duration=MAX_DURATION)
    return torch.FloatTensor(y)


def remap_path(row, data_root):
    """Remap audio path if data_root is provided (for local reproduction)."""
    if data_root:
        import os
        return os.path.join(data_root, row["diagnosis"], row["corpus"], row["filename"])
    return row["path"]


def extract_for_model(model_name, config, rows, output_dir, device, data_root=None):
    """Extract mean-pooled features for one model across all audio files."""
    out_dir = Path(output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    existing = len(list(out_dir.glob("*.npy")))
    if existing > 2000:
        print(f"  {model_name}: {existing} features already exist, skipping")
        return

    print(f"  Loading {model_name} ({config['id']})...")
    model = config["cls"].from_pretrained(config["id"]).to(device).eval()
    proc = config["proc_cls"].from_pretrained(config["id"])

    count = 0
    for i, row in enumerate(rows):
        fname = f"{row['corpus']}_{row['diagnosis']}_{row['filename']}.npy"
        if (out_dir / fname).exists():
            count += 1
            continue
        try:
            audio_path = remap_path(row, data_root)
            wav = load_audio(audio_path)
            if wav.shape[0] < TARGET_SR:  # skip < 1 second
                continue
            with torch.no_grad():
                if config["whisper_style"]:
                    inp = proc(wav.numpy(), sampling_rate=TARGET_SR, return_tensors="pt")
                    out = model.encoder(inp.input_features.to(device))
                else:
                    inp = proc(wav.numpy(), sampling_rate=TARGET_SR,
                              return_tensors="pt", padding=True)
                    out = model(inp.input_values.to(device))
                emb = out.last_hidden_state.squeeze(0).mean(0).cpu().numpy()
            np.save(out_dir / fname, emb)
            count += 1
        except Exception as e:
            continue
        if (i + 1) % 300 == 0:
            print(f"    {i+1}/{len(rows)}, extracted={count}")

    print(f"  {model_name}: {count} features total")
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Extract pretrained speech model features")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest.csv")
    parser.add_argument("--output_dir", type=str, default="features", help="Output directory")
    parser.add_argument("--model", type=str, default="all",
                        choices=list(MODEL_CONFIGS.keys()) + ["all"],
                        help="Which model to extract (default: all)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override audio root dir (e.g., /local/path/to/Original Audios)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} audio files from manifest")

    models_to_run = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    for mname in models_to_run:
        print(f"\nExtracting {mname}...")
        extract_for_model(mname, MODEL_CONFIGS[mname], rows, args.output_dir, args.device, args.data_root)

    print("\nDone.")


if __name__ == "__main__":
    main()
