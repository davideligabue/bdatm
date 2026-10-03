# Training LLMs for the Iterative Selection of Pictograms

**BDATM project 3.** Fine-tuning a language model to pick the next ARASAAC pictogram while an AAC sentence is being built, and comparing **three ways of representing that choice in the model's output**.

AAC (Augmentative and Alternative Communication) lets people who cannot rely on speech build sentences one pictogram at a time. Given the pictograms already chosen, the model proposes the next one.


| mode    | the model outputs                      | example               |
| ------- | -------------------------------------- | --------------------- |
| `id`    | the raw ARASAAC number                 | `11118`               |
| `text`  | a description, which is then looked up | `vinegar (condiment)` |
| `picto` | a dedicated new vocabulary token       | `[PICTO_11118]`       |


As in the project brief, the model sees **the pictograms chosen so far** and, for `id` and `text`, a pool of 8 candidate pictograms. Follow-up experiments: constrained decoding with a prefix trie, picto-tokens initialised from the pictogram **image**, and three models of two families.

## Results at a glance

8,467 test decisions (2,000 held-out sentences). Hit@1 on the pool of 8; all@5: the right pictogram among the top 5 of all 5,014 (picto only). Intervals are at most ±0.011.


| Hit@1              | LFM2.5-350M | Qwen3.5-0.8B | Qwen3.5-2B |
| ------------------ | ----------- | ------------ | ---------- |
| fine-tuned `id`    | 0.432       | **0.500**    | **0.508**  |
| fine-tuned `text`  | **0.444**   | 0.463        | 0.472      |
| fine-tuned `picto` | 0.391       | 0.391        | 0.392      |
| `picto`, all@5     | 0.133       | 0.131        | 0.132      |


Non-neural reference: the n-gram with back-off scores 0.387 Hit@1 and 0.108 all@5. Picto-tokens initialised from the **image** reach 0.342 on the pictograms with no text description, against 0.272 from text, on both Qwen models.

## Setup

```bash
./setup.sh                  # creates .venv and installs the pinned dependencies
source .venv/bin/activate
export HF_TOKEN=hf_...      # see "Data" below
```

Python 3.10 or newer and an NVIDIA GPU with at least 8 GB. Install the `torch` build that matches your CUDA driver first if the default one does not (see the comment at the top of `requirements.txt`).

## Running

```bash
./tools/smoke_test.sh       # ~20 minutes: every stage end to end, on tiny data
./run_all.sh --fresh        # every Qwen3.5-2B experiment, ~22 GPU-hours
./status.sh                 # what is running, and the numbers so far
```

`run_all.sh` runs every stage in order and skips a stage whose output already exists, so it can be stopped and restarted at any time. **This repository ships the predictions it produced** (that is what lets the notebook run without a GPU), so without `--fresh` every stage would be skipped; `--fresh` clears `results/`, `runs/` and `cache/` first. `MODEL=Qwen/Qwen3.5-0.8B ./run_all.sh` or `MODEL=LiquidAI/LFM2.5-350M ./run_all.sh` runs the same experiments on the other models.

Individual steps, if you prefer to run them by hand:

```bash
python -m src.data                                # build the dataset cache, print statistics
python -m src.baselines                           # non-neural baselines
python -m src.train --mode id                     # also: --mode text, --mode picto
python -m src.train --mode picto --init image     # picto-tokens from the pictogram images
python -m src.score --run id_Qwen3.5-2B           # rank the candidates, write predictions
python -m src.vision_features                     # encode the pictogram images, once
python tools/constraint_study.py                  # constraint variants
```

Options: `--model Qwen/Qwen3.5-0.8B` or `--model LiquidAI/LFM2.5-350M` for the other models, and `--seed N`. Every hyperparameter is a named constant at the top of `src/train.py` and `src/model.py`. `python -m src.score --zero-shot Qwen/Qwen3.5-2B` scores a model without fine-tuning.

## Repository structure

```
src/                        the pipeline, one file per step
  data.py                   datasets -> catalogue, sentence splits, decision steps, candidate pools
  prompts.py                prompt and target for each of the three output modes
  model.py                  4-bit loading, LoRA targets, picto-token vocabulary and initialisation
  train.py                  fine-tuning, one script for every mode
  score.py                  trained model -> ranked candidates in results/<run>/predictions.jsonl
  metrics.py                predictions -> Hit@k and MRR, strict and relaxed, with intervals
  baselines.py              random, frequency, n-gram
  constraints.py            prefix trie used for constrained decoding
  vision_features.py        pictogram images -> vectors, with the model's own vision encoder

tools/
  smoke_test.sh             ~20-minute end-to-end check on tiny data
  constraint_study.py       re-ranks the saved scores under different prefix tries (CPU)

run_all.sh                  every experiment for one model, resumable
status.sh                   progress and results so far
setup.sh                    creates the virtual environment
requirements.txt            exact versions used

notebook.ipynb              tables, plots and error analysis, reads results/ only

results/<run>/              predictions.jsonl (one ranked list per test decision)
runs/<run>/meta.json        training settings, timings and loss curve (adapters not included)
```

Run names are `<mode>_<model>`, for example `picto_text_Qwen3.5-2B` (picto mode, text initialisation); `_constrained` marks constrained decoding, `_seed43` another seed.

## Data


|                                                         |                                                         |
| ------------------------------------------------------- | ------------------------------------------------------- |
| `disi-unibo-nlp-students/ARASAAC_CommonGen_new_dataset` | 54,797 sentences, each an ordered list of pictogram ids |
| `disi-unibo-nlp-students/ARASAAC-Pictograms`            | 12,464 pictograms: keywords, categories, 500×500 PNG    |


Both datasets are gated. A valid `HF_TOKEN` is not enough: the account must have been **granted access** to both repositories. Without a connection the loader falls back to the local HuggingFace cache.

The study uses 12,000 / 1,500 / 2,000 sentences for train / validation / test (50,358 / 6,201 / 8,467 decision steps), split by sentence. Each decision offers 8 candidates: the pictograms the sentence still needs plus distractors retrieved by TF-IDF.

## Hardware

Everything ran on RTX 5060 with 8 GB: 4-bit NF4 base model, LoRA `r=16, α=32` on every attention block, `paged_adamw_8bit`, gradient checkpointing.