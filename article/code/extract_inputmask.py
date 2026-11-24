"""
Revision rerun, step 1b: region masking applied to the waveform (input-level masking).

extract_v2.py selects encoder frames after the complete recording has been encoded,
so a frame inside a pause still carries information from the surrounding speech.
Here the waveform itself is gated before it enters the encoder:

  in_speech        pauses are silenced, speech is kept
  in_pause         speech is silenced, pauses are kept
  in_shift_speech  the same speech gate, circularly shifted (wrong position, same
  in_shift_pause   size and run structure) and its complement: matched-size control

The gates are the Silero VAD segments of prepare_audio.py with a 10 ms raised-cosine
transition at every edge, so that in_speech + in_pause equals the original waveform
and no clicks are introduced. Timing is preserved: the silenced stretches stay in
place as digital silence. The encoder call is the one of extract_v2.py (same
checkpoints, final layer). Two pooled vectors are stored per condition:

  <cond>       mean over the frames of the retained region only (frames whose centre
               lies inside the kept segments). This is the speech-only / pause-only
               pooling of extract_v2.py, now with the other region silenced at the
               input, so no information from it can reach the pooled frames.
               PRIMARY analysis, fixed before any result was computed.
  <cond>_all   mean over all frames that cover the recording, silenced stretches
               included (the standard pooling). Sensitivity analysis.

The shift is the first of the K offsets used for the frame-level control
(25% to 75% of the recording), so both controls use the same offset per recording.

Usage:
  python extract_inputmask.py --work work/ --model hubert --conds main  --device mps
  python extract_inputmask.py --work work/ --model hubert --conds shift --device mps
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from extract_v2 import MODELS, MAX_SAMPLES, SR, Encoder, mean_or_zero, shift_fractions

RAMP = 160           # 10 ms transition at every gate edge
CKPT_EVERY = 200
CONDS = {"main": ["in_speech", "in_pause"], "shift": ["in_shift_speech", "in_shift_pause"]}


def speech_gate(n, segs, shift=0):
    """Hard (bool) and smoothed (float) speech gate over n samples."""
    hard = np.zeros(n, bool)
    for s, e in segs:
        hard[max(0, int(s)):min(n, int(e))] = True
    if shift:
        hard = np.roll(hard, shift)
    g = hard.astype(np.float32)
    if RAMP > 1 and n > RAMP:
        w = np.hanning(RAMP + 2)[1:-1]
        g = np.clip(np.convolve(g, w / w.sum(), mode="same"), 0.0, 1.0).astype(np.float32)
    return hard, g


def gated_pooled(enc, wav, segs, i, conds):
    """[(retained-frame mean, all-frame mean)] for the speech-kept and the pause-kept input."""
    n = len(wav)
    shift = 0 if conds == "main" else max(1, int(round(shift_fractions(i)[0] * n)))
    hard, g = speech_gate(n, segs, shift)
    out = []
    for x, keep in ((wav * g, hard), (wav * (1.0 - g), ~hard)):
        F = enc.frames(np.ascontiguousarray(x))       # same call as extract_v2.py
        Tv, centres = enc.geometry(F.shape[0], n)
        sel = keep[np.minimum(centres[:Tv].astype(int), n - 1)]
        out.append((mean_or_zero(F[:Tv], sel), F[:Tv].mean(0)))
    return out, float(hard.mean())


def run_model(name, conds, work, rows, audio, vad, device, limit=None):
    keys = [k + suf for k in CONDS[conds] for suf in ("", "_all")]
    out_dir = work / "features_inmask"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.{conds}.npz"
    part_path = out_dir / f"{name}.{conds}.partial.npz"
    if out_path.exists() and limit is None:
        print(f"{name}/{conds}: done already, skipping")
        return

    todo = [r for r in rows if r["status"] == "ok"]
    if limit:
        todo = todo[:: max(1, len(todo) // limit)][:limit]
    store = {k: [] for k in keys}
    idx, meta = [], []
    if part_path.exists() and limit is None:  # resume after an interruption
        z = np.load(part_path)
        idx, meta = z["idx"].tolist(), z["meta"].tolist()
        for k in keys:
            store[k] = list(z[k])
        print(f"{name}/{conds}: resuming after {len(idx)} files")
    done = set(idx)

    enc = Encoder(name, device)
    t0, n_new = time.time(), 0
    log = open(out_dir / f"{name}.{conds}.errors.csv", "a")

    def save(path):
        np.savez(path, idx=np.array(idx), meta=np.array(meta),
                 **{k: np.stack(v).astype(np.float32) for k, v in store.items()})

    for r in todo:
        i = int(r["idx"])
        if i in done:
            continue
        n = int(r["used_samples"])
        wav = np.array(audio[i, :n])
        try:
            pooled, frac = gated_pooled(enc, wav, vad[str(i)], i, conds)
            P = [v for pair in pooled for v in pair]       # order matches keys
            if not all(np.isfinite(v).all() for v in P):
                raise ValueError("non-finite embedding")
        except Exception as e:  # logged and excluded for every model, never silent
            log.write(f"{i},{r['filename']},{repr(e)[:200]}\n")
            log.flush()
            continue
        for k, p in zip(keys, P):
            store[k].append(p)
        idx.append(i)
        meta.append((n, frac))
        n_new += 1
        if device == "mps" and n_new % 20 == 0:
            torch.mps.empty_cache()
        if n_new % CKPT_EVERY == 0 and limit is None:
            save(part_path)
            rate = (time.time() - t0) / n_new
            left = (len(todo) - len(idx)) * rate / 60
            print(f"  {name}/{conds}: {len(idx)}/{len(todo)}  {rate:.2f} s/file  ~{left:.0f} min left", flush=True)
    log.close()
    if limit:
        print(f"{name}/{conds}: {len(idx)} files, {(time.time() - t0) / max(1, n_new):.2f} s/file (smoke test)")
        return {"idx": np.array(idx), **{k: np.stack(v) for k, v in store.items()}}
    n_err = len(todo) - len(idx)
    if n_err > max(2, 0.01 * len(todo)):
        save(part_path)
        raise SystemExit(f"{name}/{conds}: {n_err} of {len(todo)} files failed")
    save(out_path)
    if part_path.exists():
        part_path.unlink()
    print(f"{name}/{conds}: {len(idx)} files, {(time.time() - t0) / 60:.1f} min", flush=True)
    del enc
    if device == "mps":
        torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--model", default="all", choices=list(MODELS) + ["all"])
    ap.add_argument("--conds", default="main", choices=list(CONDS))
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    work = Path(args.work)

    with open(work / "manifest_v2.csv") as f:
        rows = list(csv.DictReader(f))
    audio = np.memmap(work / "audio_30s.f32", dtype="float32", mode="r",
                      shape=(len(rows), MAX_SAMPLES))
    vad = json.loads((work / "vad.json").read_text())
    for name in (list(MODELS) if args.model == "all" else [args.model]):
        run_model(name, args.conds, work, rows, audio, vad, args.device)


if __name__ == "__main__":
    main()
