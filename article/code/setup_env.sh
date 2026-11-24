#!/bin/zsh
# One-time setup for the revision reruns: conda env + model weights.
set -e
CONDA=${CONDA:-conda}
$CONDA create -y -q -n adrd-rerun python=3.11 >/dev/null
PY=$($CONDA run -n adrd-rerun python -c "import sys; print(sys.executable)")
$PY -m pip install -q --upgrade pip
$PY -m pip install -q "torch==2.5.1" "torchaudio==2.5.1" "transformers==4.46.3" "librosa==0.10.2.post1" "soundfile==0.12.1" "scikit-learn==1.5.2" "pandas==2.2.3" "silero-vad==5.1.2" "numpy<2.1"
$PY - <<'PYEOF'
import torch, transformers
print("torch", torch.__version__, "mps", torch.backends.mps.is_available(), "transformers", transformers.__version__)
from transformers import (Wav2Vec2Model, Wav2Vec2FeatureExtractor, HubertModel, WavLMModel,
                          WhisperModel, WhisperFeatureExtractor, UniSpeechSatModel)
ids = {"facebook/wav2vec2-base": Wav2Vec2Model, "facebook/wav2vec2-base-960h": Wav2Vec2Model, "facebook/hubert-base-ls960": HubertModel,
       "microsoft/wavlm-base-plus": WavLMModel, "openai/whisper-base": WhisperModel,
       "facebook/wav2vec2-xls-r-300m": Wav2Vec2Model, "openai/whisper-medium": WhisperModel,
       "microsoft/unispeech-sat-base-plus": UniSpeechSatModel}
for mid, cls in ids.items():
    m = cls.from_pretrained(mid)
    fe = (WhisperFeatureExtractor if "whisper" in mid else Wav2Vec2FeatureExtractor).from_pretrained(mid)
    print("ok", mid, sum(p.numel() for p in m.parameters())//10**6, "M params", "do_normalize=", getattr(fe, "do_normalize", None))
from silero_vad import load_silero_vad
load_silero_vad(); print("silero ok")
PYEOF
echo SETUP_DONE
