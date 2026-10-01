"""Encode pictogram images with the model's vision tower.

Produces one vector per pictogram, used to initialise picto-token embeddings.
This is the only signal available for the pictograms whose text metadata is
missing.

The tower's output width equals the text hidden size, so its pooled output can
be written straight into the embedding matrix with no extra projection.

Output: cache/vision_<model>.npz with arrays `ids` and `vectors`.

Usage:
    python -m src.vision_features --model Qwen/Qwen3.5-2B
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import PICTOGRAMS_REPO, CACHE_DIR, _load_hf, load_corpus
from .model import DEFAULT_MODEL, compute_dtype, offline_fallback

IMAGE_SIZE = 224  # 14x14 patches of size 16, merged 2x2 into 49 tokens


def _flatten(img: Image.Image) -> Image.Image:
    """Composite onto white and resize to IMAGE_SIZE.

    The PNGs have transparency; without compositing it would decode as black.
    """
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(canvas, img)
    return img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)


def cache_path(model_name: str) -> Path:
    """Where features for a given model are stored"""
    return CACHE_DIR / f"vision_{model_name.split('/')[-1]}.npz"


def load_features(model_name: str, vocab_ids: list[int]) -> dict[int, torch.Tensor]:
    """Load cached features.

    Returns:
        {pictogram id -> feature vector}
    """
    path = cache_path(model_name)
    assert path.exists(), (
        f"{path} not found -- run `python -m src.vision_features --model {model_name}` first"
    )
    blob = np.load(path)
    ids, vecs = blob["ids"], blob["vectors"]
    lookup = {int(i): torch.from_numpy(v) for i, v in zip(ids, vecs)}
    missing = [p for p in vocab_ids if p not in lookup]
    if missing:
        print(f"[vision] warning: {len(missing)} pictograms have no image feature")
    return lookup


@torch.no_grad()
def extract(model_name: str = DEFAULT_MODEL, batch_size: int = 8) -> Path:
    """Encode every pictogram in the label space and cache the vectors.

    Args:
        batch_size: images per forward pass.
    Returns:
        Path to the cache file.
    """
    # The PIL image processor avoids the torchvision dependency that
    # AutoImageProcessor and AutoProcessor pull in
    from transformers import AutoConfig
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
        Qwen2VLImageProcessorPil,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    corpus = load_corpus()
    wanted = set(corpus.vocab_ids)

    dtype = compute_dtype()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = offline_fallback(AutoConfig.from_pretrained, model_name)
    tower = Qwen3_5VisionModel._from_config(cfg.vision_config, dtype=dtype).to(device).eval()
    # Load only the vision weights out of the full checkpoint
    _load_vision_weights(tower, model_name)
    image_processor = offline_fallback(Qwen2VLImageProcessorPil.from_pretrained, model_name)

    from datasets import Image as HFImage

    ds = _load_hf(PICTOGRAMS_REPO).cast_column("image", HFImage(decode=False))

    ids: list[int] = [];  vectors: list[np.ndarray] = []
    pending_imgs: list[Image.Image] = [];  pending_ids: list[int] = []
    seen: set[int] = set()

    def flush() -> None:
        """Encode the pending batch, mean-pooling one vector per image"""
        if not pending_imgs:
            return
        inputs = image_processor(images=pending_imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
        grid = inputs["image_grid_thw"].to(device)
        out = tower(pixel_values, grid_thw=grid)
        # pooler_output holds the merged patches, whose width matches the text
        # hidden size; last_hidden_state holds the narrower raw patches
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        merge = tower.config.spatial_merge_size ** 2
        per_image = (grid[:, 0] * grid[:, 1] * grid[:, 2] // merge).tolist()
        assert feats.shape[0] == sum(per_image), (
            f"expected {sum(per_image)} merged patches, got {feats.shape[0]}"
        )
        offset = 0
        for pid, count in zip(pending_ids, per_image):
            vec = feats[offset : offset + count].float().mean(0)
            offset += count
            ids.append(pid)
            vectors.append(vec.cpu().numpy())
        pending_imgs.clear(); pending_ids.clear()

    total = len(wanted)
    for row in ds:
        pid = int(row["pictogram_id"])
        if pid not in wanted or pid in seen:
            continue
        seen.add(pid)
        raw = row["image"]
        try:
            img = _flatten(Image.open(io.BytesIO(raw["bytes"])))
        except Exception as exc:
            print(f"[vision] skipping {pid}: {exc}")
            continue
        pending_imgs.append(img); pending_ids.append(pid)
        if len(pending_imgs) >= batch_size:
            flush()
            if len(ids) % 500 < batch_size:
                print(f"  encoded {len(ids):,}/{total:,}", flush=True)
    flush()

    expected_dim = cfg.text_config.hidden_size
    assert vectors[0].shape[0] == expected_dim, (
        f"vision features are {vectors[0].shape[0]}-d but the text embedding space "
        f"is {expected_dim}-d"
    )

    out = cache_path(model_name)
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(out, ids=np.array(ids), vectors=np.stack(vectors))
    print(f"\n[vision] {len(ids):,}/{total:,} pictograms encoded, dim={vectors[0].shape[0]}")
    print(f"[vision] -> {out}")
    return out


def _load_vision_weights(tower, model_name: str) -> None:
    """Load the vision tensors out of the checkpoint into `tower`"""
    import json

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    root = Path(offline_fallback(snapshot_download, model_name,
                                 allow_patterns=["*.json", "*.safetensors"]))
    index = root / "model.safetensors.index.json"
    shards = (
        sorted({v for v in json.loads(index.read_text())["weight_map"].values()})
        if index.exists()
        else [p.name for p in root.glob("*.safetensors")]
    )

    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        for key, tensor in load_file(str(root / shard)).items():
            for prefix in ("model.visual.", "visual.", "model.vision_tower."):
                if key.startswith(prefix):
                    state[key[len(prefix) :]] = tensor
                    break
    missing, unexpected = tower.load_state_dict(state, strict=False)
    print(f"[vision] loaded {len(state)} tensors "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    assert len(state) > 0, "no vision weights found in the checkpoint"
    assert len(missing) < 5, f"vision tower is missing weights: {missing[:5]}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    extract(args.model)
