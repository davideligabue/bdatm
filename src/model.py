"""Model loading, LoRA setup and picto-token vocabulary extension.

Shared by train.py and score.py so the model is built identically at training
and evaluation time.

Qwen3.5 mixes two block types: 18 of its 24 blocks use gated linear attention
(in_proj_qkv / in_proj_z / out_proj) and 6 use softmax attention
(q_proj / k_proj / v_proj / o_proj). LoRA targets the attention projections of
both, otherwise only a quarter of the layers would be adapted.
"""

from __future__ import annotations

import torch

# LoRA target modules. in_proj_a and in_proj_b are excluded: they are small
# per-head gating projections where a rank-16 adapter would be larger than the
# layer it adapts
ATTENTION_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",      # softmax-attention blocks
    "in_proj_qkv", "in_proj_z", "out_proj",      # Qwen3.5 linear-attention blocks
    "in_proj",                                   # LFM2 convolution blocks
]

# LoRA rank, scaling and dropout, the same for every run
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

DEFAULT_MODEL = "Qwen/Qwen3.5-2B"


def offline_fallback(fn, *args, **kwargs):
    """Call a HuggingFace loader, retrying from the local cache if offline

    from_pretrained contacts the Hub even when the weights are cached, so it
    fails without a connection. The retry passes local_files_only=True, which is
    read per call; setting HF_HUB_OFFLINE here would have no effect, because
    huggingface_hub reads that variable once at import time.

    Args:
        fn: the loader to call, e.g. AutoTokenizer.from_pretrained
    Returns:
        Whatever fn returns
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        offline = any(
            k in text
            for k in ("name resolution", "ConnectError", "Connection", "Timeout",
                      "Max retries", "Offline", "502", "503")
        )
        if not offline:
            raise
        print(f"[hub] {type(exc).__name__}: reading from the local cache")
        try:
            return fn(*args, local_files_only=True, **kwargs)
        except Exception as retry_exc:
            raise RuntimeError(
                "no network and the model is not in the local cache; "
                "run once with a connection to download it"
            ) from retry_exc


def compute_dtype() -> torch.dtype:
    """bf16 when the GPU supports it natively, fp16 otherwise

    including_emulation=False matters on pre-Ampere cards, where the emulated
    path reports bf16 support but is slow and can upset the 4-bit kernels
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.bfloat16
    return torch.float16


def quantization_config(compute_dtype=torch.bfloat16):
    """4-bit NF4 config with double quantization (QLoRA)"""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_tokenizer(model_name: str = DEFAULT_MODEL):
    """Load the tokenizer with a pad token set and right padding"""
    from transformers import AutoTokenizer

    tok = offline_fallback(AutoTokenizer.from_pretrained, model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # scoring switches this to left
    return tok


def load_base_model(model_name: str = DEFAULT_MODEL):
    """Load the language model in 4-bit NF4.

    Loads only the text decoder; the vision tower is used separately by
    vision_features.py.

    Returns:
        The model, on GPU when available.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    dtype = compute_dtype()
    cfg = offline_fallback(AutoConfig.from_pretrained, model_name)
    text_cfg = getattr(cfg, "text_config", cfg)

    kwargs: dict = {
        "config": text_cfg,
        "dtype": dtype,
        "quantization_config": quantization_config(dtype),
        "device_map": {"": 0} if torch.cuda.is_available() else "cpu",
    }

    model = offline_fallback(AutoModelForCausalLM.from_pretrained, model_name, **kwargs)
    model.config.use_cache = False
    return model


# --------------------------------------------------------------------------- #
# Picto-token vocabulary
# --------------------------------------------------------------------------- #


def picto_token_strings(vocab_ids: list[int]) -> list[str]:
    """Token string for each pictogram id"""
    return [f"[PICTO_{pid}]" for pid in vocab_ids]


def add_picto_tokens(model, tokenizer, vocab_ids: list[int]) -> list[int]:
    """Add one token per pictogram and resize the embedding matrix.

    Added as ordinary tokens, not special ones, so they survive decoding.

    Returns:
        The new token ids, aligned with `vocab_ids`.
    """
    tokens = picto_token_strings(vocab_ids)
    added = tokenizer.add_tokens(tokens, special_tokens=False)
    if added:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    assert all(i is not None and i >= 0 for i in ids), "picto tokens not registered"
    assert len(set(ids)) == len(ids), "picto tokens collided with existing vocabulary"
    return ids


@torch.no_grad()
def init_picto_embeddings(
    model,
    tokenizer,
    vocab_ids: list[int],
    token_ids: list[int],
    descriptions: dict[int, str],
    init: str = "text",
    image_features: dict[int, torch.Tensor] | None = None,
) -> dict:
    """Initialise the embedding rows of the new picto tokens.

    Args:
        init: "text" to average the description's token embeddings, "image" to
            use the pooled vision-tower features, "text+image" for their sum.
        image_features: {pictogram id -> vector}, required for image inits.
    Returns:
        Stats dict with how many rows used each source.

    Rows are rescaled to the mean norm of the pretrained embeddings. The
    embedding is tied to the output layer, and an under-scaled row can never win
    the softmax.
    """
    embed = model.get_input_embeddings().weight
    device, dtype = embed.device, embed.dtype

    original = embed[: len(embed) - len(token_ids)]
    target_norm = original.float().norm(dim=1).mean().item()

    stats = {"init": init, "text_used": 0, "image_used": 0, "fallback_random": 0,
             "target_norm": target_norm}

    for pid, tid in zip(vocab_ids, token_ids):
        parts: list[torch.Tensor] = []

        if init in ("text", "text+image"):
            desc = descriptions.get(pid, "")
            desc = "" if desc.startswith("(no description") else desc
            ids = tokenizer.encode(desc, add_special_tokens=False) if desc else []
            if ids:
                vec = original[torch.tensor(ids, device=device)].float().mean(0)
                parts.append(vec / (vec.norm() + 1e-6))
                stats["text_used"] += 1

        if init in ("image", "text+image") and image_features:
            feat = image_features.get(pid)
            if feat is not None:
                vec = feat.to(device=device, dtype=torch.float32)
                parts.append(vec / (vec.norm() + 1e-6))
                stats["image_used"] += 1

        if parts:
            vec = torch.stack(parts).sum(0)
            vec = vec / (vec.norm() + 1e-6) * target_norm
        else:
            # No description and no image: random, so rows stay distinct
            vec = torch.randn(embed.shape[1], device=device, dtype=torch.float32)
            vec = vec / vec.norm() * target_norm
            stats["fallback_random"] += 1

        embed[tid] = vec.to(dtype)

    # With tied weights the output layer is the same tensor and needs no update
    out = model.get_output_embeddings()
    if out is not None and out.weight.data_ptr() != embed.data_ptr():
        out.weight[token_ids] = embed[token_ids]
        stats["untied_lm_head_synced"] = True
    return stats


# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #


def attach_lora(model, trainable_token_ids: list[int] | None = None):
    """Attach a LoRA adapter to the attention projections.

    Args:
        trainable_token_ids: train only these embedding rows, leaving the rest
            of the 248k-row matrix frozen.
    Returns:
        (peft model, list of target module names).
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    present = {n.split(".")[-1] for n, _ in model.named_modules()}
    targets = [t for t in ATTENTION_TARGETS if t in present]
    assert targets, "no LoRA target modules matched this architecture"

    kwargs: dict = dict(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    if trainable_token_ids:
        kwargs["trainable_token_indices"] = {"embed_tokens": list(trainable_token_ids)}
        kwargs["ensure_weight_tying"] = True

    peft_model = get_peft_model(model, LoraConfig(**kwargs))
    return peft_model, targets


def trainable_summary(model) -> str:
    """Return 'N trainable / M total (x%)'"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return f"{trainable:,} trainable / {total:,} total ({trainable / total:.3%})"
