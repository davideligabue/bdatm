"""Fine-tune the model for one output representation.

QLoRA: 4-bit NF4 base, LoRA r=16 on the attention projections, paged 8-bit
AdamW, gradient checkpointing. Hyperparameters are the same for every mode so
that runs stay comparable.

The prompt holds the pictograms chosen so far (plus the candidate pool for id
and text). --full-sentence adds the whole target sentence, the optimistic
upper-bound setting.

Usage:
    python -m src.train --mode id
    python -m src.train --mode text
    python -m src.train --mode picto --init text
    python -m src.train --mode picto --init image
    python -m src.train --mode picto --init text+image
    python -m src.train --mode id --model Qwen/Qwen3.5-0.8B
    python -m src.train --mode id --full-sentence
    QUICK=1 python -m src.train --mode id      # a few steps, for the smoke test

Writes runs/<name>/adapter/ and runs/<name>/meta.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, set_seed

from .data import load_corpus
from .model import (
    DEFAULT_MODEL,
    compute_dtype,
    add_picto_tokens,
    attach_lora,
    init_picto_embeddings,
    load_base_model,
    load_tokenizer,
    trainable_summary,
)
from .prompts import PROMPT_VERSION, build_prompt, build_target, render_chat

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

# Training hyperparameters, identical for every run
EPOCHS = 1
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
VAL_STEPS = 2000   # validation steps used to pick the best checkpoint
MAX_LEN = 448      # prompts measure ~237 tokens on average

# QUICK=1 trains on a handful of steps, only to check that the code runs
QUICK = os.environ.get("QUICK") == "1"


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


@dataclass
class Encoded:
    input_ids: list[int]
    labels: list[int]


def encode_step(step, corpus, mode, tokenizer, max_len: int = MAX_LEN,
                include_sentence: bool = False) -> Encoded:
    """Tokenize one decision step, supervising the answer only.

    Prompt and answer are tokenized separately so the boundary between them is
    exact and the label mask cannot drift.

    Returns:
        Encoded with input_ids and labels, where prompt positions are -100.
    """
    prompt_text = render_chat(
        build_prompt(step, corpus, mode, include_sentence), None, tokenizer, mode,
        include_sentence)
    answer_text = build_target(step, corpus, mode) + tokenizer.eos_token

    p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    # Trim the prompt from the left if needed, never the answer
    budget = max_len - len(a_ids)
    if budget < 1:
        a_ids = a_ids[: max_len - 1]
        budget = 1
    if len(p_ids) > budget:
        p_ids = p_ids[-budget:]

    return Encoded(
        input_ids=p_ids + a_ids,
        labels=[-100] * len(p_ids) + list(a_ids),
    )


class StepDataset(Dataset):
    """Tokenized decision steps for one mode"""

    def __init__(self, steps, corpus, mode, tokenizer, max_len=MAX_LEN,
                 include_sentence=False):
        self.steps, self.corpus, self.mode = steps, corpus, mode
        self.tokenizer, self.max_len = tokenizer, max_len
        self.include_sentence = include_sentence

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, i):
        enc = encode_step(self.steps[i], self.corpus, self.mode, self.tokenizer,
                          self.max_len, self.include_sentence)
        return {"input_ids": enc.input_ids, "labels": enc.labels}


class MaskedCollator:
    """Pad a batch on the left and keep the label mask.

    Left padding aligns the answer at the end of every sequence, which lets
    AnswerOnlyTrainer request logits for the last few positions only.

    The standard language-modelling collator overwrites labels with input_ids,
    so it cannot be used here; the assertion below catches that failure mode.

    Returns:
        Batch dict with input_ids, labels, attention_mask and position_ids.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id
        self._checked = False

    def __call__(self, features: list[dict]) -> dict:
        width = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention = [], [], []
        for f in features:
            pad = width - len(f["input_ids"])
            input_ids.append([self.pad_token_id] * pad + f["input_ids"])
            labels.append([-100] * pad + f["labels"])
            attention.append([0] * pad + [1] * len(f["input_ids"]))

        attn = torch.tensor(attention, dtype=torch.long)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": attn,
            # Left padding shifts absolute positions; recompute them from the mask
            "position_ids": (attn.cumsum(-1) - 1).clamp(min=0),
        }

        if not self._checked:
            self._checked = True
            n_sup = int((batch["labels"] != -100).sum())
            n_tok = int(batch["attention_mask"].sum())
            assert n_sup > 0, "no supervised positions: the label mask is empty"
            assert n_sup < n_tok, "all positions supervised: the labels were overwritten"
            assert not torch.equal(batch["labels"], batch["input_ids"]), (
                "labels == input_ids: loss masking is not in effect"
            )
            print(f"[collator] {n_sup}/{n_tok} positions supervised "
                  f"({n_sup / n_tok:.1%}), answer tokens only")
        return batch


class AnswerOnlyTrainer(Trainer):
    """Trainer that computes the loss on the answer positions only.

    The output layer is a 2048 x 248,320 matmul. Running it over every position
    would allocate a multi-gigabyte logits tensor to use ~3% of it. Because the
    collator left-pads, the answer sits at the end of the batch, so
    logits_to_keep selects exactly the needed rows.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        keep = int((labels != -100).sum(dim=1).max().item()) + 1

        outputs = model(**inputs, logits_to_keep=keep)
        logits = outputs.logits[:, :-1, :]
        target = labels[:, -(keep - 1):]

        flat_logits = logits.reshape(-1, logits.shape[-1]).float()
        flat_target = target.reshape(-1)
        if num_items_in_batch is not None:
            loss = F.cross_entropy(flat_logits, flat_target, ignore_index=-100,
                                   reduction="sum") / num_items_in_batch
        else:
            loss = F.cross_entropy(flat_logits, flat_target, ignore_index=-100)
        return (loss, outputs) if return_outputs else loss


def _length_report(dataset: StepDataset, n: int = 500) -> dict:
    """Token-length statistics over the first `n` examples"""
    lens = [len(dataset[i]["input_ids"]) for i in range(min(n, len(dataset)))]
    lens.sort()
    return {
        "mean": sum(lens) / len(lens),
        "p50": lens[len(lens) // 2],
        "p95": lens[int(len(lens) * 0.95)],
        "max": lens[-1],
        "over_max_len": sum(x >= MAX_LEN for x in lens) / len(lens),
    }


# --------------------------------------------------------------------------- #


def run_name_for(mode: str, init: str, model: str, seed: int, full_sentence: bool) -> str:
    """Folder name under runs/, e.g. picto_text_Qwen3.5-2B or id_Qwen3.5-2B_fullsent"""
    name = mode + (f"_{init.replace('+', '')}" if mode == "picto" else "")
    name += f"_{model.split('/')[-1]}"
    if seed != 42:
        name += f"_seed{seed}"
    if full_sentence:
        name += "_fullsent"
    return ("_smoke_" + name) if QUICK else name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=["id", "text", "picto"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--init", default="text", choices=["text", "image", "text+image"],
                    help="picto-token initialisation (picto mode only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full-sentence", action="store_true",
                    help="also show the whole target sentence (upper-bound setting)")
    args = ap.parse_args()

    # Must come before the model is built: LoRA's A matrix and the random
    # fallback rows for pictograms without a description are both drawn from the
    # global torch RNG, and Trainer only seeds it later, in its constructor
    set_seed(args.seed)

    run_name = run_name_for(args.mode, args.init, args.model, args.seed, args.full_sentence)
    out_dir = RUNS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== run: {run_name} -> {out_dir}")

    corpus = load_corpus()
    tokenizer = load_tokenizer(args.model)
    model = load_base_model(args.model)

    picto_token_ids = None
    init_stats = None
    if args.mode == "picto":
        picto_token_ids = add_picto_tokens(model, tokenizer, corpus.vocab_ids)
        image_features = None
        if "image" in args.init:
            from .vision_features import load_features

            image_features = load_features(args.model, corpus.vocab_ids)
        init_stats = init_picto_embeddings(
            model, tokenizer, corpus.vocab_ids, picto_token_ids,
            corpus.description, init=args.init, image_features=image_features,
        )
        print(f"[picto] {len(picto_token_ids):,} tokens added; init={init_stats}")

    model, targets = attach_lora(model, trainable_token_ids=picto_token_ids)
    print(f"[lora] targets={targets}")
    print(f"[lora] {trainable_summary(model)}")

    train_steps = corpus.steps["train"][:480] if QUICK else corpus.steps["train"]
    val_steps = corpus.steps["val"][: 64 if QUICK else VAL_STEPS]
    train_ds = StepDataset(train_steps, corpus, args.mode, tokenizer,
                           include_sentence=args.full_sentence)
    val_ds = StepDataset(val_steps, corpus, args.mode, tokenizer,
                         include_sentence=args.full_sentence)

    lengths = _length_report(train_ds)
    print(f"[data] train={len(train_ds):,} val={len(val_ds):,} | token lengths {lengths}")

    total_optim_steps = int(len(train_ds) * EPOCHS / BATCH_SIZE)
    eval_every = max(50, total_optim_steps // 3)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=EPOCHS,
        max_steps=30 if QUICK else -1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        # Only eval_loss is used; without this the Trainer would keep the logits
        # of the whole validation split in memory
        prediction_loss_only=True,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=eval_every,
        save_strategy="steps",
        save_steps=eval_every,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = AnswerOnlyTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,          # validation split, never test
        data_collator=MaskedCollator(tokenizer.pad_token_id),
    )

    started = time.time()
    trainer.train()
    elapsed = time.time() - started

    trainer.save_model(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))

    if picto_token_ids:
        # The adapter stores a delta on top of each base embedding row, so the
        # initialisation has to be saved separately to reload the run
        embed = trainer.model.get_input_embeddings()
        base = getattr(embed, "original_module", embed)
        rows = base.weight.detach()[picto_token_ids].to(torch.float32).cpu()
        torch.save(
            {"token_ids": picto_token_ids, "vocab_ids": corpus.vocab_ids, "rows": rows},
            out_dir / "adapter" / "picto_init.pt",
        )
        print(f"[picto] saved init rows {tuple(rows.shape)} -> adapter/picto_init.pt")

    meta = {
        "run_name": run_name,
        "mode": args.mode,
        "model": args.model,
        "init": args.init if args.mode == "picto" else None,
        "init_stats": init_stats,
        "lora_targets": targets,
        "trainable": trainable_summary(model),
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "seed": args.seed,
        "include_sentence": args.full_sentence,
        "prompt_version": PROMPT_VERSION,
        "dtype": str(compute_dtype()),
        "effective_batch": BATCH_SIZE,
        "train_steps": len(train_ds),
        "val_steps": len(val_ds),
        "batch_size": BATCH_SIZE,
        "token_lengths": lengths,
        "train_seconds": elapsed,
        "n_picto_tokens": len(picto_token_ids) if picto_token_ids else 0,
        "log_history": trainer.state.log_history,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\n=== done in {elapsed / 3600:.2f} h -> {out_dir}/adapter")


if __name__ == "__main__":
    main()
