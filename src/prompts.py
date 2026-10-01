"""Prompt and target construction for the three output representations.

    id     -> "11118"                   the ARASAAC id
    text   -> "vinegar (condiment)"     a description, resolved by lookup
    picto  -> "[PICTO_11118]"           a dedicated vocabulary token

`id` and `text` use the same prompt and differ only in the target, so comparing
them isolates the output representation. `picto` drops the candidate pool and
writes the already-selected pictograms as tokens.

By default the prompt holds what the project brief lists as input: the
pictograms chosen so far (and, for id and text, the candidate pool). The target
sentence is added only in the full-sentence setting, an optimistic upper bound.
"""

from __future__ import annotations

import re

from .data import NO_DESCRIPTION, Corpus

MODES = ("id", "text", "picto")

# Version 1: the wording every full-sentence run was trained with. Kept
# unchanged so those runs are scored with exactly their training prompt
SYSTEM_PROMPT = {
    "id": (
        "You help build sentences with ARASAAC pictograms. "
        "Given the target sentence, the pictograms already chosen and a list of "
        "candidates, reply with the id of the next pictogram. Reply with the id only."
    ),
    "text": (
        "You help build sentences with ARASAAC pictograms. "
        "Given the target sentence, the pictograms already chosen and a list of "
        "candidates, reply with the description of the next pictogram. "
        "Reply with the description only."
    ),
    "picto": (
        "You help build sentences with ARASAAC pictograms. "
        "Given the target sentence and the pictograms already chosen, "
        "reply with the token of the next pictogram."
    ),
}

# Version 2: the same instructions without the sentence, for the default setting
SYSTEM_PROMPT_NO_SENTENCE = {
    "id": (
        "You help build sentences with ARASAAC pictograms. "
        "Given the pictograms already chosen and a list of candidates, "
        "reply with the id of the next pictogram. Reply with the id only."
    ),
    "text": (
        "You help build sentences with ARASAAC pictograms. "
        "Given the pictograms already chosen and a list of candidates, "
        "reply with the description of the next pictogram. "
        "Reply with the description only."
    ),
    "picto": (
        "You help build sentences with ARASAAC pictograms. "
        "Given the pictograms already chosen, "
        "reply with the token of the next pictogram."
    ),
}

PROMPT_VERSION = 2  # recorded in meta.json by train.py


def system_prompt(mode: str, include_sentence: bool, version: int = PROMPT_VERSION) -> str:
    """System prompt for a mode and setting

    Runs trained before version 2 (meta.json without "prompt_version") used
    version 1 for every setting, and are scored with it
    """
    if include_sentence or version < 2:
        return SYSTEM_PROMPT[mode]
    return SYSTEM_PROMPT_NO_SENTENCE[mode]

_PICTO_RE = re.compile(r"\[PICTO_(\d+)\]")


def picto_token(pictogram_id: int) -> str:
    """Vocabulary token for a pictogram"""
    return f"[PICTO_{pictogram_id}]"


def parse_picto_token(text: str) -> int | None:
    """Return the id of the first picto token in `text`, or None.

    Generation often continues past the answer, so only the first token counts.
    """
    match = _PICTO_RE.search(text)
    return int(match.group(1)) if match else None


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


def _prefix_line(step: dict, corpus: Corpus) -> str:
    """Render the already-selected pictograms as text"""
    if not step["prefix_ids"]:
        return "(nothing yet)"
    return " | ".join(corpus.description.get(p, NO_DESCRIPTION) for p in step["prefix_ids"])


def build_prompt(step: dict, corpus: Corpus, mode: str,
                 include_sentence: bool = False) -> str:
    """Build the user prompt for one decision step

    Args:
        step: a decision step from data.build_steps
        mode: "id", "text" or "picto"
        include_sentence: add the full target sentence (upper-bound setting)
    Returns:
        The prompt text
    """
    sentence_line = f"Sentence: {step['sentence']}\n" if include_sentence else ""

    if mode in ("id", "text"):
        candidates = "\n".join(
            f"{pid} {desc}" for pid, desc in corpus.pool_descriptions(step["pool_ids"])
        )
        return (
            f"{sentence_line}"
            f"Selected: {_prefix_line(step, corpus)}\n"
            f"Candidates:\n{candidates}\n"
            f"Next:"
        )

    if mode == "picto":
        prefix = " ".join(picto_token(p) for p in step["prefix_ids"]) or "(nothing yet)"
        return f"{sentence_line}Selected: {prefix}\nNext:"

    raise ValueError(f"unknown mode {mode!r}")


def build_target(step: dict, corpus: Corpus, mode: str) -> str:
    """Build the supervised target for one decision step"""
    gold = step["gold_id"]
    if mode == "id":
        return str(gold)
    if mode == "text":
        return corpus.description.get(gold, NO_DESCRIPTION)
    if mode == "picto":
        return picto_token(gold)
    raise ValueError(f"unknown mode {mode!r}")


def candidate_answer(pictogram_id: int, corpus: Corpus, mode: str) -> str:
    """Build the string the model would emit if this candidate were the answer.

    Scoring ranks candidates by the log-probability of these strings.
    """
    if mode == "id":
        return str(pictogram_id)
    if mode == "text":
        return corpus.description.get(pictogram_id, NO_DESCRIPTION)
    if mode == "picto":
        return picto_token(pictogram_id)
    raise ValueError(f"unknown mode {mode!r}")


# --------------------------------------------------------------------------- #
# Chat formatting
# --------------------------------------------------------------------------- #


def render_chat(prompt: str, target: str | None, tokenizer, mode: str,
                include_sentence: bool = False, version: int = PROMPT_VERSION) -> str:
    """Apply the chat template, with thinking disabled.

    Args:
        target: the answer for training, or None to stop at the assistant turn
            for generation and scoring.
        include_sentence, version: select the system prompt, see system_prompt
    Returns:
        The formatted string.
    """
    messages = [
        {"role": "system", "content": system_prompt(mode, include_sentence, version)},
        {"role": "user", "content": prompt},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        text = tokenizer.apply_chat_template(
            **{**kwargs, "conversation": messages, "enable_thinking": False}
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    if target is None:
        return text
    return text + target + tokenizer.eos_token


def _demo() -> None:
    """Print one decision step rendered in all three modes"""
    from .data import load_corpus

    corpus = load_corpus()
    step = corpus.steps["test"][1]
    for mode in MODES:
        print("=" * 72)
        print(f"MODE: {mode}")
        print("-" * 72)
        print(system_prompt(mode, include_sentence=False))
        print(build_prompt(step, corpus, mode))
        print(f"  --> TARGET: {build_target(step, corpus, mode)!r}")


if __name__ == "__main__":
    _demo()
