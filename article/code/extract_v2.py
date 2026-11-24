"""
Revision rerun, step 1: one encoder pass per model and condition, all pooled variants.

Same checkpoints, final layer and mean pooling as the paper. For every usable
recording and model it stores mean-pooled final-layer vectors:
  full_all       all encoder frames. For Whisper this includes the frames of the
                 zero-padded tail up to 30 s (the pooling used in the paper)
  full_valid     only frames that cover real audio (same as full_all for SSL models)
  speech, pause  valid frames whose centre lies inside / outside a Silero VAD
                 speech segment. Frame times come from the 20 ms encoder hop, so
                 the masks are aligned for Whisper inputs shorter than 30 s (the
                 paper mapped VAD times proportionally onto all 1,500 frames)
  shift_speech, shift_pause
                 K circularly shifted copies of the speech mask (same size and run
                 structure, wrong position): matched-size random-mask control
  rev_all, rev_valid     same pooling on the time-reversed waveform
  lrev_all, lrev_valid   same pooling after reversing every 50 ms window in place
                 (keeps most linguistic content, adds a similar kind of signal
                 manipulation): control for generic sensitivity to unnatural input

Usage:
  python extract_v2.py --work work/ --model all --device mps
"""
import argparse
import csv
import json
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from transformers import (Wav2Vec2Model, Wav2Vec2FeatureExtractor, HubertModel, WavLMModel,
                          WhisperModel, WhisperFeatureExtractor, UniSpeechSatModel)

warnings.filterwarnings("ignore")

SR = 16000
MAX_SAMPLES = SR * 30
HOP = 320            # 20 ms per encoder frame for all seven models
SSL_RF = 400         # receptive field of the wav2vec2-style conv encoder (samples)
K_SHIFT = 5
LREV_WIN = 800       # 50 ms
CKPT_EVERY = 200

MODELS = {
    "wav2vec2_ssl":   ("facebook/wav2vec2-base", Wav2Vec2Model, False),  # pretrained only, added in revision
    "wav2vec2":       ("facebook/wav2vec2-base-960h", Wav2Vec2Model, False),  # + CTC ASR fine-tuning
    "hubert":         ("facebook/hubert-base-ls960", HubertModel, False),
    "wavlm":          ("microsoft/wavlm-base-plus", WavLMModel, False),
    "whisper_base":   ("openai/whisper-base", WhisperModel, True),
    "xlsr":           ("facebook/wav2vec2-xls-r-300m", Wav2Vec2Model, False),
    "whisper_medium": ("openai/whisper-medium", WhisperModel, True),
    "unispeech_sat":  ("microsoft/unispeech-sat-base-plus", UniSpeechSatModel, False),
}
FULL_KEYS = ["full_all", "full_valid", "speech", "pause", "rev_all", "rev_valid",
             "lrev_all", "lrev_valid"]


def local_reverse(w, win=LREV_WIN):
    out = w.copy()
    m = len(w) // win * win
    out[:m] = w[:m].reshape(-1, win)[:, ::-1].reshape(-1)
    return out


def shift_fractions(i):
    """Same offsets for every model: fraction of the valid length in [0.25, 0.75]."""
    rng = np.random.default_rng(1000 + i)
    return rng.uniform(0.25, 0.75, size=K_SHIFT)


class _CPUConv(torch.nn.Module):
    """Runs one conv layer on the CPU. On MPS the first conv of the wav2vec2-style
    encoders fails for 30 s inputs (> 65,536 output steps); the rest stays on MPS."""

    def __init__(self, conv):
        super().__init__()
        self.conv = conv.to("cpu")

    def forward(self, x):
        return self.conv(x.to("cpu")).to(x.device)


class Encoder:
    def __init__(self, name, device):
        mid, cls, self.whisper = MODELS[name]
        model = cls.from_pretrained(mid)
        if self.whisper:
            model = model.encoder
            self.fe = WhisperFeatureExtractor.from_pretrained(mid)
        else:
            self.fe = Wav2Vec2FeatureExtractor.from_pretrained(mid)
        self.model = model.to(device).eval()
        if device == "mps" and not self.whisper:
            first = self.model.feature_extractor.conv_layers[0]
            first.conv = _CPUConv(first.conv)
        self.device = device
        self.dim = model.config.d_model if self.whisper else model.config.hidden_size

    @torch.no_grad()
    def frames(self, wav):
        if self.whisper:
            x = self.fe(wav, sampling_rate=SR, return_tensors="pt").input_features
            h = self.model(x.to(self.device)).last_hidden_state
        else:
            x = self.fe(wav, sampling_rate=SR, return_tensors="pt").input_values
            h = self.model(x.to(self.device)).last_hidden_state
        return h[0].float().cpu().numpy()

    def geometry(self, T, n):
        """Number of frames that cover real audio, and the centre of every frame (samples)."""
        t = np.arange(T)
        if self.whisper:
            return min(T, math.ceil(n / HOP)), HOP * t + HOP / 2
        return T, HOP * t + SSL_RF / 2


def mean_or_zero(F, mask):
    return F[mask].mean(0) if mask.any() else np.zeros(F.shape[1], dtype=np.float32)


def pool_all(enc, wav, segs, fracs):
    F = enc.frames(wav)
    T = F.shape[0]
    Tv, centres = enc.geometry(T, len(wav))
    valid = np.zeros(T, bool)
    valid[:Tv] = True
    speech = np.zeros(T, bool)
    for s, e in segs:
        speech |= (centres >= s) & (centres < e)
    speech &= valid
    pause = valid & ~speech
    out = {
        "full_all": F.mean(0), "full_valid": F[:Tv].mean(0),
        "speech": mean_or_zero(F, speech), "pause": mean_or_zero(F, pause),
    }
    ss, sp = [], []
    for u in fracs:
        m = np.zeros(T, bool)
        m[:Tv] = np.roll(speech[:Tv], max(1, int(round(u * Tv))))
        ss.append(mean_or_zero(F, m))
        sp.append(mean_or_zero(F, valid & ~m))
    out["shift_speech"], out["shift_pause"] = np.stack(ss), np.stack(sp)
    meta = (T, Tv, int(speech.sum()), int(pause.sum()))

    R = enc.frames(np.ascontiguousarray(wav[::-1]))
    out["rev_all"], out["rev_valid"] = R.mean(0), R[:Tv].mean(0)
    L = enc.frames(local_reverse(wav))
    out["lrev_all"], out["lrev_valid"] = L.mean(0), L[:Tv].mean(0)
    return out, meta


def run_model(name, work, rows, audio, vad, device):
    out_path = work / "features_v2" / f"{name}.npz"
    part_path = work / "features_v2" / f"{name}.partial.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"{name}: done already, skipping")
        return

    todo = [r for r in rows if r["status"] == "ok"]
    store = {k: [] for k in FULL_KEYS + ["shift_speech", "shift_pause"]}
    idx, meta = [], []
    if part_path.exists():  # resume after an interruption
        z = np.load(part_path)
        idx = z["idx"].tolist()
        meta = z["meta"].tolist()
        for k in store:
            store[k] = list(z[k])
        print(f"{name}: resuming after {len(idx)} files")
    done = set(idx)

    enc = Encoder(name, device)
    t0, n_new = time.time(), 0
    log = open(work / "features_v2" / f"{name}.errors.csv", "a")

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
            pooled, m = pool_all(enc, wav, vad[str(i)], shift_fractions(i))
        except Exception as e:  # logged and excluded for every model, never silent
            log.write(f"{i},{r['filename']},{repr(e)[:200]}\n")
            log.flush()
            continue
        for k in store:
            store[k].append(pooled[k])
        idx.append(i)
        meta.append(m)
        n_new += 1
        if device == "mps" and n_new % 20 == 0:
            torch.mps.empty_cache()  # long runs otherwise exhaust MPS memory (seen with XLS-R)
        if n_new % CKPT_EVERY == 0:
            save(part_path)
            rate = (time.time() - t0) / n_new
            left = (len(todo) - len(idx)) * rate / 60
            print(f"  {name}: {len(idx)}/{len(todo)}  {rate:.2f} s/file  ~{left:.0f} min left", flush=True)
    log.close()
    n_err = len(todo) - len(idx)
    if n_err > max(2, 0.01 * len(todo)):
        save(part_path)
        raise SystemExit(f"{name}: {n_err} of {len(todo)} files failed, see {name}.errors.csv")
    save(out_path)
    if part_path.exists():
        part_path.unlink()
    print(f"{name}: {len(idx)} files, {(time.time() - t0) / 60:.1f} min", flush=True)
    del enc
    if device == "mps":
        torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--model", default="all", choices=list(MODELS) + ["all"])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    work = Path(args.work)

    with open(work / "manifest_v2.csv") as f:
        rows = list(csv.DictReader(f))
    audio = np.memmap(work / "audio_30s.f32", dtype="float32", mode="r",
                      shape=(len(rows), MAX_SAMPLES))
    vad = json.loads((work / "vad.json").read_text())
    names = list(MODELS) if args.model == "all" else [args.model]
    for name in names:
        run_model(name, work, rows, audio, vad, args.device)


if __name__ == "__main__":
    main()
