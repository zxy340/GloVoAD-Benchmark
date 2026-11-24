"""
Revision rerun, step 0: manifest, participant IDs, durations, decoded audio cache and VAD.

Reads "Original Audios/<Diagnosis>/<Corpus>/<file>" (same layout as
data_preparation/build_manifest.py in the public repo), decodes every file once
(librosa, 16 kHz mono, first 30 s, as in the paper), runs Silero VAD once with
the paper's settings, and writes into --out:
  manifest_v2.csv   one row per file: status, durations, participant ID, VAD summary, md5
  audio_30s.f32     float32 memmap [n_files, 480000], zero padded, row = manifest row
  vad.json          speech segments in samples for every usable file
  data_report.txt   counts by class x corpus x language, duplicate and ID checks

Usage:
  python prepare_audio.py --data_root "/path/to/Original Audios" --out work/
"""
import argparse
import csv
import hashlib
import json
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import librosa
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

warnings.filterwarnings("ignore")

SR = 16000
MAX_DUR = 30
MAX_SAMPLES = SR * MAX_DUR
MIN_SAMPLES = SR  # the paper skips recordings shorter than 1 s
DIAGNOSES = ["Control", "MCI", "AD", "ProbableAD", "PPA"]
AUDIO_EXT = (".mp3", ".wav", ".flac")

# Language per corpus. The public repo listed VAS as Spanish and Lu as Mandarin;
# the GloVoAD draft (Spanish n = 360, Mandarin n = 43, Mandarin Lu used only as an
# external set) implies VAS and Lu are English. Marked for confirmation by Wei Bo.
CORPUS_LANGUAGE = {
    "Pitt": "English", "Delaware": "English", "WLS": "English",
    "Baycrest": "English", "Hopkins": "English", "DePaul": "English",
    "VAS": "English", "Lu": "English",
    "Ivanova": "Spanish", "Ye": "Mandarin",
}

# Participant ID rules (regex on the file stem, group 1 = participant).
# Pitt: "001-0" -> visit 0 of participant 001. The diagnosis is NOT part of the
# ID any more, so a participant whose diagnosis changed across visits stays in
# one group. Other corpora default to one participant per file until their
# naming scheme is confirmed (see the prefix report in data_report.txt).
PARTICIPANT_RULES = {
    "Pitt": r"^(\d+)-",                 # 001-0, 001-1: yearly visits of participant 001
    "Hopkins": r"^([A-Z]+\d+)[a-z]$",   # AMR6628a: session letter after the participant code
}
# Checked on the GloVoAD "Original Audios" release (2,066 files): every other corpus
# has one file per participant with a unique stem, and no stem occurs under two
# diagnoses except Pitt participants 172 (Control, MCI) and 212 (MCI, ProbableAD).


def participant_id(corpus, filename):
    stem = Path(filename).stem
    rule = PARTICIPANT_RULES.get(corpus)
    if rule:
        m = re.match(rule, stem)
        if m:
            return f"{corpus}_{m.group(1)}"
    return f"{corpus}_{stem}"


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def scan(data_root):
    rows = []
    for diag in DIAGNOSES:
        ddir = data_root / diag
        if not ddir.exists():
            continue
        for cdir in sorted(p for p in ddir.iterdir() if p.is_dir()):
            for f in sorted(cdir.iterdir()):
                if f.suffix.lower() in AUDIO_EXT:
                    rows.append({
                        "path": str(f), "diagnosis": diag, "corpus": cdir.name,
                        "language": CORPUS_LANGUAGE.get(cdir.name, "Unknown"),
                        "filename": f.name,
                        "participant": participant_id(cdir.name, f.name),
                    })
    return rows


def write_report(rows, out):
    ok = [r for r in rows if r["status"] == "ok"]
    lines = []
    lines.append(f"Files found: {len(rows)}   usable: {len(ok)}")
    for st, c in Counter(r["status"] for r in rows).items():
        lines.append(f"  status {st}: {c}")

    def table(title, key_fn, items):
        lines.append(f"\n{title}")
        cnt = defaultdict(Counter)
        for r in items:
            cnt[key_fn(r)][r["diagnosis"]] += 1
        lines.append(f"  {'':22s}" + "".join(f"{d:>11s}" for d in DIAGNOSES) + f"{'Total':>9s}")
        for k in sorted(cnt):
            lines.append(f"  {k:22s}" + "".join(f"{cnt[k][d]:11d}" for d in DIAGNOSES)
                         + f"{sum(cnt[k].values()):9d}")
        lines.append(f"  {'Total':22s}" + "".join(
            f"{sum(cnt[k][d] for k in cnt):11d}" for d in DIAGNOSES) + f"{len(items):9d}")

    table("Usable recordings by corpus (language)", lambda r: f"{r['corpus']} ({r['language']})", ok)
    table("Usable recordings by language", lambda r: r["language"], ok)

    # participants: count each participant once per class
    lines.append("\nUnique participants among usable recordings, by class")
    per_class = defaultdict(set)
    for r in ok:
        per_class[r["diagnosis"]].add(r["participant"])
    for d in DIAGNOSES:
        lines.append(f"  {d:12s} {len(per_class[d])}")
    allp = {r["participant"] for r in ok}
    lines.append(f"  {'all':12s} {len(allp)}")
    multi = Counter(r["participant"] for r in ok)
    nm = [p for p, c in multi.items() if c > 1]
    lines.append(f"  participants with >1 recording: {len(nm)} "
                 f"({sum(multi[p] for p in nm)} recordings, "
                 f"{100 * sum(multi[p] for p in nm) / max(len(ok), 1):.1f}% of usable)")

    # participants that appear under more than one diagnosis
    diag_of = defaultdict(set)
    for r in ok:
        diag_of[r["participant"]].add(r["diagnosis"])
    cross = {p: sorted(d) for p, d in diag_of.items() if len(d) > 1}
    lines.append(f"\nParticipants listed under more than one diagnosis: {len(cross)}")
    for p, d in list(cross.items())[:30]:
        lines.append(f"  {p}: {', '.join(d)}")

    # exact duplicate files
    by_hash = defaultdict(list)
    for r in rows:
        by_hash[r["md5"]].append(f"{r['diagnosis']}/{r['corpus']}/{r['filename']}")
    dups = [v for v in by_hash.values() if len(v) > 1]
    lines.append(f"\nByte-identical duplicate files: {len(dups)} groups")
    for v in dups[:30]:
        lines.append("  " + " | ".join(v))

    # naming patterns, to decide participant rules for non-Pitt corpora
    lines.append("\nFile naming per corpus (to check for several recordings per participant)")
    by_corpus = defaultdict(list)
    for r in rows:
        by_corpus[r["corpus"]].append(Path(r["filename"]).stem)
    for c, stems in sorted(by_corpus.items()):
        prefixes = {re.split(r"[-_.\s]", s)[0] for s in stems}
        lines.append(f"  {c:10s} files={len(stems):5d} distinct_prefix={len(prefixes):5d} "
                     f"examples: {', '.join(stems[:6])}")

    # durations
    lines.append("\nOriginal duration (s) of usable files, by class: median [IQR], % shorter than 30 s")
    for d in DIAGNOSES:
        du = np.array([float(r["orig_dur_s"]) for r in ok if r["diagnosis"] == d])
        if len(du):
            q = np.percentile(du, [25, 50, 75])
            lines.append(f"  {d:12s} {q[1]:7.1f} [{q[0]:.1f}, {q[2]:.1f}]  "
                         f"{100 * np.mean(du < MAX_DUR):5.1f}%")
    du = np.array([float(r["orig_dur_s"]) for r in ok])
    if len(du):
        lines.append(f"  {'all':12s} {np.median(du):7.1f}  {100 * np.mean(du < MAX_DUR):5.1f}% shorter than 30 s, "
                     f"{100 * np.mean(du > MAX_DUR):5.1f}% longer (truncated)")
        sf = np.array([float(r["speech_frac"]) for r in ok])
        lines.append(f"  VAD speech fraction of the analysed window: median {np.median(sf):.2f}")

    (out / "data_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="path to 'Original Audios'")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_root, out = Path(args.data_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = scan(data_root)
    print(f"Found {len(rows)} audio files")

    audio = np.memmap(out / "audio_30s.f32", dtype="float32", mode="w+",
                      shape=(len(rows), MAX_SAMPLES))
    vad_model = load_silero_vad()
    vad = {}
    for i, r in enumerate(rows):
        r["md5"] = md5(r["path"])
        r.update({"status": "ok", "orig_dur_s": "", "used_samples": 0,
                  "n_speech_segments": 0, "speech_s": 0.0, "speech_frac": 0.0, "error": ""})
        try:
            r["orig_dur_s"] = f"{librosa.get_duration(path=r['path']):.3f}"
            y, _ = librosa.load(r["path"], sr=SR, mono=True, duration=MAX_DUR)
        except Exception as e:  # recorded, not silently skipped
            r["status"], r["error"] = "load_error", repr(e)[:200]
            continue
        n = min(len(y), MAX_SAMPLES)
        r["used_samples"] = n
        if n < MIN_SAMPLES:
            r["status"] = "too_short"
            continue
        audio[i, :n] = y[:n]
        ts = get_speech_timestamps(torch.from_numpy(np.ascontiguousarray(y[:n])), vad_model,
                                   sampling_rate=SR, threshold=0.5,
                                   min_speech_duration_ms=100, min_silence_duration_ms=100)
        segs = [[int(t["start"]), int(t["end"])] for t in ts]
        vad[str(i)] = segs
        sp = sum(e - s for s, e in segs)
        r["n_speech_segments"] = len(segs)
        r["speech_s"] = round(sp / SR, 3)
        r["speech_frac"] = round(sp / n, 4)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(rows)}")
    audio.flush()

    fields = ["idx", "path", "diagnosis", "corpus", "language", "filename", "participant",
              "status", "error", "orig_dur_s", "used_samples", "n_speech_segments",
              "speech_s", "speech_frac", "md5"]
    with open(out / "manifest_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"idx": i, **{k: r.get(k, "") for k in fields if k != "idx"}})
    (out / "vad.json").write_text(json.dumps(vad))
    write_report(rows, out)


if __name__ == "__main__":
    main()
