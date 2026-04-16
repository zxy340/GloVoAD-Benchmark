"""
Step 0: Build a CSV manifest of all audio files with diagnosis and corpus labels.

Usage:
    python build_manifest.py --data_root /path/to/GloVoAD/Original\ Audios --output manifest.csv

Output columns: path, diagnosis, corpus, language, filename
"""
import os
import csv
import argparse
from pathlib import Path
from collections import Counter

CORPUS_LANGUAGE = {
    "Pitt": "English",
    "Delaware": "English",
    "WLS": "English",
    "Baycrest": "English",
    "Hopkins": "English",
    "VAS": "Spanish",
    "Ivanova": "Spanish",
    "Lu": "Mandarin",
    "Ye": "Mandarin",
    "DePaul": "English",
}

DIAGNOSES = ["AD", "Control", "MCI", "PPA", "ProbableAD"]


def build_manifest(data_root, output_path):
    data_root = Path(data_root)
    rows = []

    for diagnosis in DIAGNOSES:
        diag_dir = data_root / diagnosis
        if not diag_dir.exists():
            continue
        for corpus_dir in sorted(diag_dir.iterdir()):
            if not corpus_dir.is_dir():
                continue
            corpus = corpus_dir.name
            lang = CORPUS_LANGUAGE.get(corpus, "Unknown")
            for audio_file in sorted(corpus_dir.iterdir()):
                if audio_file.suffix.lower() in (".mp3", ".wav", ".flac"):
                    rows.append({
                        "path": str(audio_file),
                        "diagnosis": diagnosis,
                        "corpus": corpus,
                        "language": lang,
                        "filename": audio_file.name,
                    })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "diagnosis", "corpus", "language", "filename"]
        )
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    print(f"Total files: {len(rows)}")
    print("\nBy diagnosis:")
    for d, c in sorted(Counter(r["diagnosis"] for r in rows).items()):
        print(f"  {d}: {c}")
    print("\nBy corpus:")
    for co, c in sorted(Counter(r["corpus"] for r in rows).items()):
        print(f"  {co}: {c}")
    print("\nBy language:")
    for l, c in sorted(Counter(r["language"] for r in rows).items()):
        print(f"  {l}: {c}")
    print(f"\nManifest saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build audio manifest CSV")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to GloVoAD 'Original Audios' directory")
    parser.add_argument("--output", type=str, default="manifest.csv",
                        help="Output CSV path")
    args = parser.parse_args()
    build_manifest(args.data_root, args.output)
