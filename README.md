# Matching Speech Models to ADRD Subtypes: A Mechanistic Benchmark

Code for reproducing all experiments in the paper.

## Overview

We benchmark 7 pretrained speech models on 5-class ADRD subtype classification (Control / MCI / AD / ProbableAD / PPA) using the GloVoAD dataset, then use two verification experiments to explain *why* different models excel at different subtypes:

- **Region Masking**: Reveals that denoising-trained models (WavLM) depend on pause regions for AD/MCI detection
- **Linguistic Perturbation**: Reveals that ASR-trained models (Whisper) depend on linguistic content for PPA detection

## Models

| Model | Pretraining Paradigm | Checkpoint | Dim |
|-------|---------------------|-----------|-----|
| Wav2Vec2 | Contrastive SSL | `facebook/wav2vec2-base-960h` | 768 |
| HuBERT | Cluster Prediction | `facebook/hubert-base-ls960` | 768 |
| WavLM | Cluster + Denoising | `microsoft/wavlm-base-plus` | 768 |
| Whisper-base | ASR Supervised | `openai/whisper-base` | 512 |
| XLS-R | Multilingual SSL | `facebook/wav2vec2-xls-r-300m` | 1024 |
| Whisper-medium | ASR Supervised (Large) | `openai/whisper-medium` | 1024 |
| UniSpeech-SAT | Speaker-Aware SSL | `microsoft/unispeech-sat-base-plus` | 768 |

## Directory Structure

```
├── data_preparation/
│   └── build_manifest.py          # Step 0: Build CSV manifest from audio files
├── feature_extraction/
│   └── extract_features.py        # Step 1: Extract frozen features from 7 models
├── benchmark/
│   └── run_benchmark.py           # Step 2: 5-class benchmark (Table 3)
├── handcrafted_baseline/
│   └── run_handcrafted_baseline.py # Step 2b: Librosa feature baseline (Table 3)
├── region_masking/
│   └── run_region_masking.py      # Step 3: Region masking experiment (Figure 2)
├── linguistic_perturbation/
│   └── run_linguistic_perturbation.py # Step 4: Audio reversal experiment (Table 5, Figure 3)
├── figures/
│   └── generate_figures.py        # Generate all 4 paper figures
├── requirements.txt
└── README.md
```

## Reproduction Steps

### Prerequisites

```bash
pip install -r requirements.txt
```

The GloVoAD dataset should be organized as:
```
Original Audios/
├── AD/
│   ├── Pitt/
│   └── ...
├── Control/
├── MCI/
├── PPA/
└── ProbableAD/
```

### Step 0: Build Manifest

```bash
python data_preparation/build_manifest.py \
    --data_root "/path/to/GloVoAD/Original Audios" \
    --output manifest.csv
```

### Step 1: Extract Features

```bash
# Extract all 7 models (requires GPU, ~2-3 hours)
python feature_extraction/extract_features.py \
    --manifest manifest.csv \
    --output_dir features/ \
    --device cuda:0

# Or extract one model at a time
python feature_extraction/extract_features.py \
    --manifest manifest.csv \
    --model whisper_medium \
    --device cuda:0
```

### Step 2: Run Benchmark (Table 3)

```bash
# Pretrained model benchmark
python benchmark/run_benchmark.py \
    --manifest manifest.csv \
    --feat_dir features/ \
    --device cuda:0

# Handcrafted feature baseline
python handcrafted_baseline/run_handcrafted_baseline.py \
    --manifest manifest.csv
```

### Step 3: Region Masking (Figure 2)

```bash
# Run for all models (requires GPU, ~4-6 hours)
python region_masking/run_region_masking.py \
    --manifest manifest.csv \
    --device cuda:0

# Or run for one model
python region_masking/run_region_masking.py \
    --manifest manifest.csv \
    --model wavlm \
    --device cuda:0
```

### Step 4: Linguistic Perturbation (Table 5, Figure 3)

```bash
# Run for all models (requires GPU, ~4-6 hours)
python linguistic_perturbation/run_linguistic_perturbation.py \
    --manifest manifest.csv \
    --device cuda:0
```

### Generate Figures

```bash
python figures/generate_figures.py --output_dir figures/output/
```

## Evaluation Protocol

- **Cross-validation**: 5-fold StratifiedGroupKFold × 3 random seeds (15 folds total)
- **Speaker independence**: Same speaker's multiple recordings never appear in both train and test
- **Classifier**: 2-layer MLP (256 hidden units, ReLU, 30% dropout)
- **Training**: Adam optimizer, lr=1e-3, weight_decay=1e-4, 80 epochs, batch_size=64
- **Class balancing**: Inverse-frequency class weights in cross-entropy loss
- **Metric**: Macro F1 score (primary), per-class F1 scores
