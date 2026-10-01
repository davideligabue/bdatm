"""Score a trained model and save its rankings.

For every decision step the model assigns a log-probability to each candidate,
the candidates are ranked, and the ranking is written to
results/<name>/predictions.jsonl. Metrics are computed separately by metrics.py.

Per step the file records:
    ranked_ids    the 8 pool candidates, ranked by per-token log-probability
                  (id, text) or total log-probability (picto)
    scores        the score each candidate was ranked by
    open_ranked   ranking over the whole label space (picto mode)
    generated     the free-form greedy output and the id parsed from it

Each run is scored with the prompt it was trained on (setting and prompt
version are read from its meta.json).

Usage:
    python -m src.score --run id_Qwen3.5-2B
    python -m src.score --run picto_text_Qwen3.5-2B --constrain
    python -m src.score --zero-shot Qwen/Qwen3.5-2B
    python -m src.score --zero-shot Qwen/Qwen3.5-2B --full-sentence
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch

from .data import NO_DESCRIPTION, load_corpus
from .model import load_base_model, load_tokenizer, picto_token_strings
from .prompts import (PROMPT_VERSION, build_prompt, candidate_answer, parse_picto_token,
                      render_chat)

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"

MAX_ANSWER_TOKENS = 48
BATCH_STEPS = 4    # decision steps per forward pass, each expands to 8 candidates
GEN_BATCH = 16     # prompts per batch for open-set ranking and generation

# QUICK=1 scores the first 100 test steps, only to check that the code runs
QUICK = os.environ.get("QUICK") == "1"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_run(run_dir: Path, corpus):
    """Rebuild the model a training run produced.

    Returns:
        (model, tokenizer, meta dict, picto token ids or None)
    """
    from peft import PeftModel

    meta = json.loads((run_dir / "meta.json").read_text())
    adapter = run_dir / "adapter"

    tokenizer = load_tokenizer(meta["model"])
    model = load_base_model(meta["model"])

    picto_token_ids = None
    if meta["mode"] == "picto":
        tokens = picto_token_strings(corpus.vocab_ids)
        tokenizer.add_tokens(tokens, special_tokens=False)
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        picto_token_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The adapter holds a delta on each row, so restore the saved
        # initialisation before applying it
        init_path = adapter / "picto_init.pt"
        assert init_path.exists(), f"missing {init_path}; retrain or copy it over"
        blob = torch.load(init_path, map_location="cpu")
        assert blob["vocab_ids"] == corpus.vocab_ids, "vocabulary changed since training"
        embed = model.get_input_embeddings().weight
        with torch.no_grad():
            embed[picto_token_ids] = blob["rows"].to(embed.device, embed.dtype)

    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    model.config.use_cache = True
    return model, tokenizer, meta, picto_token_ids


def load_zero_shot(model_name: str, full_sentence: bool):
    """Load the base model with no adapter, for the no-fine-tuning baseline (id mode)"""
    tokenizer = load_tokenizer(model_name)
    model = load_base_model(model_name)
    model.eval()
    model.config.use_cache = True
    meta = {"mode": "id", "model": model_name, "init": None, "zero_shot": True,
            "include_sentence": full_sentence, "prompt_version": PROMPT_VERSION}
    return model, tokenizer, meta, None


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


@torch.no_grad()
def score_continuations(
    model, tokenizer, prompts: list[str], answers: list[str], device
) -> tuple[list[float], list[int]]:
    """Log-probability of each answer given its prompt.

    Sequences are left-padded so all answers end at the same position, allowing
    logits_to_keep to request only the final positions.

    Args:
        prompts: one prompt per candidate (repeated across a step's candidates).
        answers: the candidate strings.
    Returns:
        (total log-probability per answer, token count per answer)
    """
    p_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
    a_ids = [tokenizer(a, add_special_tokens=False)["input_ids"][:MAX_ANSWER_TOKENS]
             for a in answers]

    full = [p + a for p, a in zip(p_ids, a_ids)]
    width = max(len(f) for f in full)
    pad_id = tokenizer.pad_token_id

    input_ids, attn, labels = [], [], []
    for f, a in zip(full, a_ids):
        pad = width - len(f)
        input_ids.append([pad_id] * pad + f)
        attn.append([0] * pad + [1] * len(f))
        labels.append([-100] * (width - len(a)) + a)

    input_ids = torch.tensor(input_ids, device=device)
    attn = torch.tensor(attn, device=device)
    labels = torch.tensor(labels, device=device)
    position_ids = (attn.cumsum(-1) - 1).clamp(min=0)

    keep = max(len(a) for a in a_ids) + 1
    logits = model(
        input_ids=input_ids, attention_mask=attn,
        position_ids=position_ids, logits_to_keep=keep,
    ).logits

    tgt = labels[:, -(keep - 1):]
    mask = tgt != -100

    # Accumulate one position at a time: a full log_softmax over
    # (batch, positions, 248k vocab) would allocate several GB
    totals = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.float32)
    for i in range(keep - 1):
        step_logits = logits[:, i, :].float()
        picked = step_logits.gather(-1, tgt[:, i].clamp(min=0).unsqueeze(-1)).squeeze(-1)
        totals += (picked - torch.logsumexp(step_logits, dim=-1)) * mask[:, i]
    return totals.tolist(), [len(a) for a in a_ids]


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: list[str], device, max_new_tokens=24) -> list[str]:
    """Greedy generation for a batch of prompts.

    Decodes with skip_special_tokens=False so picto tokens survive.

    Returns:
        The generated text per prompt, with chat markers stripped.
    """
    tokenizer.padding_side = "left"
    batch = tokenizer(prompts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(device)
    out = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new = out[:, batch["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(new, skip_special_tokens=False)
    cleaned = []
    for t in texts:
        for special in (tokenizer.eos_token, tokenizer.pad_token, "<|im_end|>", "<|endoftext|>"):
            if special:
                t = t.replace(special, "")
        cleaned.append(t.strip())
    return cleaned


def parse_generation(text: str, mode: str, corpus, description_index=None) -> int | None:
    """Extract a pictogram id from generated text.

    Generation often continues past the answer, so the first match is taken.

    Returns:
        The pictogram id, or None if nothing valid was produced.
    """
    if mode == "picto":
        return parse_picto_token(text)
    if mode == "id":
        match = re.search(r"\d+", text)
        if not match:
            return None
        pid = int(match.group(0))
        return pid if pid in corpus.description else None
    if mode == "text":
        if description_index is None:
            return None
        return description_index.get(text.strip().lower())
    return None


# --------------------------------------------------------------------------- #


def run_scoring(model, tokenizer, meta, picto_token_ids, corpus,
                constrain: bool = False) -> list[dict]:
    """Score every decision step of the test split.

    Args:
        constrain: suppress candidates the prefix trie disallows.
    Returns:
        One prediction record per decision step.
    """
    device = next(model.parameters()).device
    mode = meta["mode"]
    batch_steps, gen_batch = BATCH_STEPS, GEN_BATCH
    # Score with the prompt the run was trained on. Runs from before these fields
    # existed were all trained with the full sentence and prompt version 1
    keep_sentence = meta.get("include_sentence", True)
    version = meta.get("prompt_version", 1)
    steps = corpus.steps["test"][:100] if QUICK else corpus.steps["test"]

    description_index = {d.strip().lower(): p for p, d in corpus.description.items()
                         if d != NO_DESCRIPTION}

    trie = None
    if constrain:
        from .constraints import PrefixTrie

        trie = PrefixTrie.from_steps(corpus.steps["train"])

    open_token_tensor = None
    if mode == "picto":
        open_token_tensor = torch.tensor(picto_token_ids, device=device)

    records: list[dict] = []
    started = time.time()

    for start in range(0, len(steps), batch_steps):
        chunk = steps[start : start + batch_steps]

        # One (prompt, candidate) pair per candidate of every step in the chunk
        prompts, answers = [], []
        for step in chunk:
            prompt = render_chat(build_prompt(step, corpus, mode, keep_sentence),
                                 None, tokenizer, mode, keep_sentence, version)
            for pid in step["pool_ids"]:
                prompts.append(prompt)
                answers.append(candidate_answer(pid, corpus, mode))

        tokenizer.padding_side = "right"
        totals, lengths = score_continuations(model, tokenizer, prompts, answers, device)

        per_step: list[list[tuple[int, float, float]]] = [[] for _ in chunk]
        cursor = 0
        for si, step in enumerate(chunk):
            for pid in step["pool_ids"]:
                total, n_tok = totals[cursor], max(1, lengths[cursor])
                per_step[si].append((pid, total, total / n_tok))
                cursor += 1

        for si, step in enumerate(chunk):
            scored = per_step[si]
            if trie is not None:
                allowed = trie.allowed(step["prefix_ids"], {p for p, _, _ in scored})
                scored = [
                    (p, t + (0.0 if p in allowed else -1e4), m + (0.0 if p in allowed else -1e4))
                    for p, t, m in scored
                ]
            by_mean = [p for p, _, _ in sorted(scored, key=lambda x: -x[2])]
            by_sum = [p for p, _, _ in sorted(scored, key=lambda x: -x[1])]
            # id and text are ranked by the per-token average, so long descriptions
            # are not penalised. picto is ranked by the total: a picto answer is one
            # token, and a pool candidate outside the picto vocabulary would
            # otherwise be scored as an average over the sub-word pieces of
            # "[PICTO_<id>]", which the model can never emit as an answer
            primary = 1 if mode == "picto" else 2
            records.append(
                {
                    "step_id": step["step_id"],
                    "gold_id": step["gold_id"],
                    "t": step["t"],
                    "n_pictos": step["n_pictos"],
                    "has_description": corpus.description.get(step["gold_id"], NO_DESCRIPTION)
                    != NO_DESCRIPTION,
                    "pool_ids": step["pool_ids"],
                    "ranked_ids": by_sum if mode == "picto" else by_mean,
                    "ranked_ids_sum": by_sum,   # total log-probability
                    "scores": {str(x[0]): round(x[primary], 4) for x in scored},
                }
            )

        if start % (batch_steps * 25) == 0:
            done = min(start + batch_steps, len(steps))
            rate = done / max(1e-6, time.time() - started)
            eta = (len(steps) - done) / max(1e-6, rate) / 60
            print(f"  scored {done:,}/{len(steps):,}  ({rate:.1f} steps/s, ETA {eta:.1f} min)",
                  flush=True)

    if mode == "picto":
        _open_rank_picto(model, tokenizer, corpus, steps, records, open_token_tensor,
                         device, mode, gen_batch, trie, keep_sentence, version)

    _generate_all(model, tokenizer, corpus, steps, records, device, mode,
                  gen_batch, description_index, keep_sentence, version)

    return records


@torch.no_grad()
def _open_rank_picto(model, tokenizer, corpus, steps, records, token_tensor,
                     device, mode, batch, trie, keep_sentence, version):
    """Rank the whole label space from one forward pass per step.

    Only possible in picto mode, where each pictogram is a single token.
    Writes "open_ranked" (top 20) into each record.
    """
    print("  open-set ranking over the full picto vocabulary ...")
    vocab = corpus.vocab_ids
    for start in range(0, len(steps), batch):
        chunk = steps[start : start + batch]
        prompts = [render_chat(build_prompt(s, corpus, mode, keep_sentence),
                               None, tokenizer, mode, keep_sentence, version)
                   for s in chunk]
        tokenizer.padding_side = "left"
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        logits = model(**enc, position_ids=pos, logits_to_keep=1).logits[:, -1, :].float()
        picto_logits = logits[:, token_tensor]
        if trie is not None:
            for i, step in enumerate(chunk):
                allowed = trie.allowed(step["prefix_ids"], set(vocab))
                mask = torch.tensor([p not in allowed for p in vocab], device=device)
                picto_logits[i] = picto_logits[i].masked_fill(mask, -1e4)
        top = picto_logits.topk(k=min(20, len(vocab)), dim=-1).indices.tolist()
        for i, row in enumerate(top):
            records[start + i]["open_ranked"] = [vocab[j] for j in row]


@torch.no_grad()
def _generate_all(model, tokenizer, corpus, steps, records, device, mode,
                  batch, description_index, keep_sentence, version):
    """Run greedy generation and store the raw text plus the parsed id"""
    print("  free-form greedy generation ...")
    max_new = {"id": 12, "text": 32, "picto": 6}[mode]
    for start in range(0, len(steps), batch):
        chunk = steps[start : start + batch]
        prompts = [render_chat(build_prompt(s, corpus, mode, keep_sentence),
                               None, tokenizer, mode, keep_sentence, version)
                   for s in chunk]
        texts = generate_batch(model, tokenizer, prompts, device, max_new_tokens=max_new)
        for i, text in enumerate(texts):
            rec = records[start + i]
            rec["generated"] = text
            rec["generated_id"] = parse_generation(text, mode, corpus, description_index)


# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None, help="folder name under runs/")
    ap.add_argument("--constrain", action="store_true",
                    help="suppress candidates the prefix trie disallows")
    ap.add_argument("--zero-shot", default=None, metavar="MODEL",
                    help="score this base model without fine-tuning (id mode)")
    ap.add_argument("--full-sentence", action="store_true",
                    help="with --zero-shot: also show the target sentence")
    args = ap.parse_args()
    assert args.run or args.zero_shot, "pass --run or --zero-shot"

    corpus = load_corpus()

    if args.zero_shot:
        model, tokenizer, meta, picto_ids = load_zero_shot(args.zero_shot, args.full_sentence)
        name = f"zeroshot_id_{args.zero_shot.split('/')[-1]}"
        name += "_fullsent" if args.full_sentence else ""
    else:
        model, tokenizer, meta, picto_ids = load_run(RUNS / args.run, corpus)
        name = args.run + ("_constrained" if args.constrain else "")
    if QUICK and not name.startswith("_smoke_"):
        name = "_smoke_" + name

    print(f"=== scoring {name}  (mode={meta['mode']})")
    started = time.time()
    records = run_scoring(model, tokenizer, meta, picto_ids, corpus, constrain=args.constrain)
    elapsed = time.time() - started

    out_dir = RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.jsonl").open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    (out_dir / "scoring_meta.json").write_text(json.dumps(
        {**{k: v for k, v in meta.items() if k != "log_history"},
         "split": "test", "n": len(records), "constrained": args.constrain,
         "seconds": elapsed}, indent=2, default=str))

    from .metrics import compute, table

    m = compute(records, corpus.equivalence)
    print(f"\n{table({name: m})}")
    print(f"\nscored {len(records):,} steps in {elapsed / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
