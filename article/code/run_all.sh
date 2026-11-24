#!/bin/zsh
# Revision rerun: feature extraction in two parallel processes, then all analyses.
# Usage: ./run_all.sh <work dir>   (prepare_audio.py must have been run on it)
RR=${0:A:h}
PY=${PY:-python}   # set PY to the Python of the adrd-rerun environment
WORK=${1:-$RR/work}
mkdir -p $RR/logs
( for m in whisper_medium wav2vec2_ssl wav2vec2 hubert; do
    $PY $RR/extract_v2.py --work $WORK --model $m --device ${DEVICE:-mps}; done ) > $RR/logs/extract_A_${WORK:t}.log 2>&1 &
( for m in xlsr wavlm unispeech_sat whisper_base; do
    $PY $RR/extract_v2.py --work $WORK --model $m --device ${DEVICE:-mps}; done ) > $RR/logs/extract_B_${WORK:t}.log 2>&1 &
wait
$PY $RR/evaluate_v2.py --work $WORK --jobs 8 --boot 2000 > $RR/logs/evaluate_${WORK:t}.log 2>&1
echo "ALL_DONE $(date)"
