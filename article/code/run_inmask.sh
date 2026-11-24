#!/bin/zsh
# Input-level region masking: feature extraction in two parallel processes.
# Usage: ./run_inmask.sh [work dir]   (runs the main conditions, then the shifted-gate control)
RR=${0:A:h}
PY=${PY:-python}   # set PY to the Python of the adrd-rerun environment
WORK=${1:-$RR/work}
mkdir -p $RR/logs
cd $RR
for PH in main shift; do
  ( for m in whisper_medium wav2vec2_ssl wav2vec2 hubert; do
      $PY $RR/extract_inputmask.py --work $WORK --model $m --conds $PH --device ${DEVICE:-mps}; done ) > $RR/logs/inmask_${PH}_A.log 2>&1 &
  ( for m in xlsr wavlm unispeech_sat whisper_base; do
      $PY $RR/extract_inputmask.py --work $WORK --model $m --conds $PH --device ${DEVICE:-mps}; done ) > $RR/logs/inmask_${PH}_B.log 2>&1 &
  wait
  echo "INMASK_${PH}_DONE $(date)"
done
