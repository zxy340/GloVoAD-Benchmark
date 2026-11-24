# Code and results for "Matching Speech Models to ADRD Subtypes: An Exploratory Benchmark"

This folder holds the code that produced every result reported in the article (*Information*, MDPI, 2026), together with the result files. It replaces the earlier code in the top-level folders of this repository, which belongs to the version of this work presented at ICIBM 2026.

## Contents

| Path | What it is |
|---|---|
| `code/prepare_audio.py` | Reads the GloVoAD audio, builds the manifest and participant IDs, decodes the first 30 s of each file at 16 kHz, runs Silero VAD |
| `code/extract_v2.py` | Frozen final-layer embeddings for the eight checkpoints, with the full, speech-only, pause-only, shifted-mask, reversed and locally reversed pooled vectors |
| `code/evaluate_v2.py` | All classifiers, baselines, bootstrap confidence intervals, paired tests, corpus identification and within-corpus analyses |
| `code/make_tables_figures.py` | Builds the LaTeX tables and PDF figures of the paper from the results folder |
| `code/run_all.sh` | Runs extraction for all models, then the evaluation |
| `code/setup_env.sh`, `requirements.txt` | Environment (Python 3.11, PyTorch 2.5.1, transformers 4.46.3) |
| `results/` | The outputs of `evaluate_v2.py` used in the paper |

## Data

The recordings come from DementiaBank and cannot be redistributed. DementiaBank membership can be requested at talkbank@cmu.edu. The scripts expect the GloVoAD layout `Original Audios/<Diagnosis>/<Corpus>/<file>`.

## Running

Run the commands from this folder, and give the shell scripts the work directory as an absolute path. The shell scripts use zsh.

```bash
CONDA=conda ./code/setup_env.sh
PY=/path/to/envs/adrd-rerun/bin/python
$PY code/prepare_audio.py --data_root "/path/to/Original Audios" --out work/
PY=$PY DEVICE=mps ./code/run_all.sh "$PWD/work"      # DEVICE=cuda or cpu elsewhere
$PY code/make_tables_figures.py --results work/results_v2 --figures figures --tables tables
```

On an Apple M1 Pro (16 GB), extraction took about 1.5 h per model (6 min for Whisper-base), run as two parallel processes, and the evaluation took about 15 min. On MPS, the first convolution of the self-supervised models runs on the CPU because MPS does not support its 30 s input length.

## Results files

- `summary.csv`: F1 for every experiment, condition, model and class. `point` is F1 on the pooled out-of-fold predictions averaged over the three seeds. `lo` and `hi` are the 95% participant-level bootstrap CI (2,000 resamples). `fold_mean` and `fold_sd` are over the 15 test folds.
- `paired.csv`: paired bootstrap differences (A minus B) with CI, two-sided p-value and Holm-adjusted p-value within each family and class.
- `corpus_identification.csv`, `within_corpus.csv`, `confusion.json`, `run_info.json`: corpus-level analyses, confusion matrices and sample counts.
- `oof_predictions.csv.gz`: the predicted class of every recording for every seed and every experiment, condition and model (columns named `experiment|condition|model`), with its test fold. Recordings are identified as `<Corpus>/<file name>`. **Diagnostic labels are not included.** DementiaBank members can recover them from the folder structure of the GloVoAD release and recompute every reported result.

Experiment names: `bench` trains on full vectors and tests on every condition (full training in the article), `bench_all` uses Whisper pooling over the padded 30 s window, and `tt_*` trains and tests on the same manipulated vectors (matched training in the article).

## Input-level region masking

The frame-level masking of `extract_v2.py` selects encoder frames after the whole recording has been encoded. The scripts below repeat the analysis with the mask applied to the waveform before encoding (Sections 3.4 and 4.4, Tables A3 and A4, triangles in Figure 2).

| Path | What it is |
|---|---|
| `code/extract_inputmask.py` | Replaces speech segments or pauses by digital silence in place (10 ms transitions), encodes each input, and stores the mean over the frames of the retained region (`in_speech`, `in_pause`) and over all frames (`*_all`). `--conds shift` does the same with the circularly shifted mask (`in_shift_speech`, `in_shift_pause`) |
| `code/run_inmask.sh` | Runs both extractions for the eight checkpoints in two parallel processes |
| `code/evaluate_inputmask.py` | Classifiers, bootstrap CIs and paired tests for the pooled task and within corpora, with the folds, seeds and resamples of `evaluate_v2.py` |
| `code/supp_inputmask.py` | Corpus identification from the speech-only and pause-only vectors, and the test of whether the speech-minus-pause gap differs between models |
| `code/report_inputmask.py` | Text report of every number quoted in the article |
| `code/make_tables_figures_inmask.py` | Figure 2, Tables A3 and A4 and the numbers quoted in the text |
| `results_inmask/` | The outputs of the scripts above |

```bash
PY=$PY DEVICE=mps ./code/run_inmask.sh "$PWD/work"
$PY code/evaluate_inputmask.py --work work --jobs 8 --boot 2000
$PY code/supp_inputmask.py --work work --res results_inmask
$PY code/report_inputmask.py --work work
$PY code/make_tables_figures_inmask.py --work work --fig figures --gen tables
```

These scripts import `extract_v2.py` and `evaluate_v2.py`, so they must be run after `run_all.sh` on the same work directory. `evaluate_inputmask.py` checks that its full-representation predictions are identical to those of the benchmark run. On an Apple M1 Pro the two extractions took about 4 h each and the evaluation about 40 min.

Files in `results_inmask/`: `summary.csv`, `paired.csv` and `oof_predictions.csv.gz` have the layout described above (again without diagnostic labels). `within_corpus.csv` and `within_paired.csv` hold the same analysis inside each corpus. `corpus_identification_inmask.csv` gives corpus identification from each kind of vector, and `gap_between_models.csv` the paired comparison of the speech-minus-pause gap between models. The pooling over the retained frames was fixed as the primary analysis before any result was computed, and the pooling over all frames (`*_all`) is a sensitivity analysis.
