"""
Stage 3 (Optional): Multimodal Embeddings with CLIP / SigLIP
=============================================================
For products that have an image_url, downloads the image and encodes it
with CLIP (openai/clip-vit-base-patch32) alongside the text.

Two fusion strategies are supported:
  - text_only  : encode only full_text (same space as CLIP text encoder)
  - multimodal : average of text + image embeddings (rows with no image
                 fall back to text-only)

Saves:
  {stem}_embeddings_clip_text.npy        – CLIP text embeddings (all rows)
  {stem}_embeddings_clip_multimodal.npy  – fused text+image (all rows)
  {stem}_clip_metadata.parquet           – row index, asin, has_image flag

Usage:
    python stage3_multimodal.py --input office_products_micro.parquet
    python stage3_multimodal.py --input office_products_micro.parquet --max-images 1000
"""

import argparse
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
EMBED_DIM = 512          # CLIP ViT-B/32 output dim
IMAGE_TIMEOUT = 5        # seconds per HTTP request
DEFAULT_BATCH_SIZE = 64
MAX_IMAGES_DEFAULT = None  # None = process all rows that have an image_url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows from {path}")
    return df


def fetch_image(url: str) -> Image.Image | None:
    """Download and return a PIL image, or None on any error."""
    try:
        resp = requests.get(url, timeout=IMAGE_TIMEOUT, stream=True)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def encode_texts_clip(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: list[str],
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Encode text with CLIP text encoder, return L2-normalised (N, D) array."""
    all_embs = []
    for start in tqdm(range(0, len(texts), batch_size), desc="CLIP text"):
        batch = texts[start : start + batch_size]
        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,          # CLIP hard limit
        ).to(device)
        with torch.no_grad():
            embs = model.get_text_features(**inputs)
            embs = embs / embs.norm(dim=-1, keepdim=True)  # L2 norm
        all_embs.append(embs.cpu().numpy())
    return np.vstack(all_embs).astype(np.float32)


def encode_images_clip(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_urls: list[str | None],
    batch_size: int,
    device: str,
    max_images: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Download and encode images from URLs using proper GPU batching.

    Returns:
        image_embs  : (N, D) array — zero vector if image unavailable
        has_image   : (N,) bool array
    """
    n = len(image_urls)
    image_embs = np.zeros((n, EMBED_DIM), dtype=np.float32)
    has_image = np.zeros(n, dtype=bool)
    images_processed = 0

    # Шаг 1: Итерируемся по датасету окнами размера batch_size
    for start in tqdm(range(0, n, batch_size), desc="Downloading + encoding images"):
        if max_images is not None and images_processed >= max_images:
            break

        batch_urls = image_urls[start : start + batch_size]
        valid_images = []
        valid_indices = []

        # Шаг 2: Скачиваем картинки текущего батча
        for j, url in enumerate(batch_urls):
            global_idx = start + j
            if not url:
                continue
            if max_images is not None and images_processed >= max_images:
                break

            img = fetch_image(url)
            if img is not None:
                valid_images.append(img)
                valid_indices.append(global_idx)
                images_processed += 1

        # Шаг 3: Если в батче есть успешные скачивания, отправляем их в CLIP ПАЧКОЙ
        if valid_images:
            # padding не нужен и не поддерживается для изображений: CLIPImageProcessor
            # уже приводит все картинки к фиксированному размеру 224×224 (resize + crop),
            # поэтому все тензоры в батче и так одной формы.
            inputs = processor(images=valid_images, return_tensors="pt").to(device)
            with torch.no_grad():
                embs = model.get_image_features(**inputs)
                embs = embs / embs.norm(dim=-1, keepdim=True)  # L2 norm

            # Раскладываем эмбеддинги по своим исходным индексам в общей матрице
            image_embs[valid_indices] = embs.cpu().numpy()
            has_image[valid_indices] = True

    print(f"Successfully encoded {images_processed:,} images out of {n:,} rows")
    return image_embs, has_image


def fuse_embeddings(
    text_embs: np.ndarray,
    image_embs: np.ndarray,
    has_image: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Weighted average fusion:
      - rows with image  → alpha*text + (1-alpha)*image  (re-normalised)
      - rows without     → text only
    """
    fused = text_embs.copy()
    idx = np.where(has_image)[0]
    if len(idx) == 0:
        return fused

    blended = alpha * text_embs[idx] + (1 - alpha) * image_embs[idx]
    norms = np.linalg.norm(blended, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    fused[idx] = (blended / norms).astype(np.float32)
    return fused


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Multimodal CLIP embeddings")
    parser.add_argument("--input", default="office_products_micro.parquet")
    parser.add_argument("--output-dir", default="embeddings")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-images",
        type=int,
        default=MAX_IMAGES_DEFAULT,
        help="Cap on number of images to download (useful for testing)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Text weight in text+image fusion (0.0=image only, 1.0=text only)",
    )
    parser.add_argument("--model", default=CLIP_MODEL_NAME)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- Load data ----
    df = load_data(args.input)
    texts = df["full_text"].fillna("").tolist()
    texts = [t if t.strip() else " " for t in texts]

    image_urls: list[str | None] = (
        df["image_url"].where(df["image_url"].notna(), None).tolist()
        if "image_url" in df.columns
        else [None] * len(df)
    )

    n_with_url = sum(1 for u in image_urls if u)
    print(f"Rows with image_url: {n_with_url:,} / {len(df):,}")

    # ---- Load CLIP ----
    print(f"\nLoading CLIP model: {args.model}")
    t_load = time.perf_counter()
    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(device)
    model.eval()
    print(f"Model loaded in {time.perf_counter() - t_load:.1f}s")

    # ---- Text embeddings ----
    print("\nEncoding texts with CLIP text encoder...")
    t0 = time.perf_counter()
    text_embs = encode_texts_clip(model, processor, texts, args.batch_size, device)
    text_elapsed = time.perf_counter() - t0
    print(f"Text encoding: {len(texts)/text_elapsed:.0f} texts/s")

    # ---- Image embeddings ----
    print("\nDownloading and encoding images...")
    t0 = time.perf_counter()
    image_embs, has_image = encode_images_clip(
        model, processor, image_urls, args.batch_size, device, args.max_images
    )
    img_elapsed = time.perf_counter() - t0

    # ---- Fuse ----
    print(f"\nFusing text + image (alpha={args.alpha})...")
    multimodal_embs = fuse_embeddings(text_embs, image_embs, has_image, alpha=args.alpha)

    # ---- Save ----
    stem = Path(args.input).stem

    text_path = output_dir / f"{stem}_embeddings_clip_text.npy"
    np.save(text_path, text_embs)
    print(f"Saved CLIP text embeddings   → {text_path}")

    mm_path = output_dir / f"{stem}_embeddings_clip_multimodal.npy"
    np.save(mm_path, multimodal_embs)
    print(f"Saved multimodal embeddings  → {mm_path}")

    meta_path = output_dir / f"{stem}_clip_metadata.parquet"
    meta_cols = [c for c in ["parent_asin", "title", "price", "image_url"] if c in df.columns]
    meta_df = df[meta_cols].copy()
    meta_df["has_image"] = has_image
    meta_df.to_parquet(meta_path, index=False)
    print(f"Saved CLIP metadata          → {meta_path}")

    # ---- Summary ----
    print(f"\n{'='*55}")
    print(f"  Stage 3 complete")
    print(f"  Text embs shape      : {text_embs.shape}")
    print(f"  Multimodal embs shape: {multimodal_embs.shape}")
    print(f"  Images encoded       : {has_image.sum():,}")
    print(f"  Text encoding speed  : {len(texts)/text_elapsed:.0f} texts/s")
    
    # Защита от деления на ноль, если ни одна картинка не обработалась
    img_speed = has_image.sum() / img_elapsed if img_elapsed > 0 else 0
    print(f"  Image download+encode: {img_speed:.1f} imgs/s")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()