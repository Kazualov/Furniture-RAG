"""
Embedding Generation Pipeline
=======================================
Loads office_products_micro.parquet (or full), encodes `full_text`
with all-MiniLM-L6-v2, and saves vectors to disk as .npy + metadata.

Usage:
    python generate_embeddings.py --input office_products_micro.parquet
    python generate_embeddings.py --input office_products_full.parquet --batch-size 256
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 128
EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows from {path}")
    print(f"Columns: {list(df.columns)}")
    return df


def get_texts(df: pd.DataFrame) -> list[str]:
    """
    Primary field: full_text (rich concatenated text built in data prep).
    Fallback: title only, if full_text is missing/empty.
    """
    if "full_text" in df.columns:
        texts = df["full_text"].fillna("").tolist()
    else:
        print("Warning: 'full_text' column not found, falling back to 'title'")
        texts = df["title"].fillna("").tolist()

    # Replace empty strings with a single space so the model never gets an
    # empty input (SentenceTransformers handles whitespace gracefully).
    texts = [t if t.strip() else " " for t in texts]
    return texts


def encode_in_batches(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    device: str,
) -> np.ndarray:
    """
    Encodes texts batch-by-batch and returns a (N, D) float32 array.
    Uses tqdm for a live progress bar.
    """
    all_embeddings = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch = texts[start : start + batch_size]
        with torch.no_grad():
            embs = model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,   # L2-norm → cosine sim == dot product
                device=device,
            )
        all_embeddings.append(embs)

    return np.vstack(all_embeddings).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings")
    parser.add_argument(
        "--input",
        default="office_products_micro.parquet",
        help="Path to input .parquet file",
    )
    parser.add_argument(
        "--output-dir",
        default="embeddings",
        help="Directory where outputs are saved",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Encoding batch size (increase for GPU, decrease for low RAM)",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="HuggingFace model name or local path",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- Load data ----
    df = load_parquet(args.input)
    texts = get_texts(df)
    print(f"Texts to encode: {len(texts):,}")

    # ---- Load model ----
    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model, device=device)
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # ---- Encode ----
    t0 = time.perf_counter()
    embeddings = encode_in_batches(model, texts, args.batch_size, device)
    elapsed = time.perf_counter() - t0

    n = len(texts)
    print(f"\nEncoding done: {n:,} vectors in {elapsed:.1f}s  ({n/elapsed:.0f} texts/s)")
    print(f"Embedding matrix shape: {embeddings.shape}  dtype: {embeddings.dtype}")

    # ---- Save embeddings ----
    stem = Path(args.input).stem
    emb_path = output_dir / f"{stem}_embeddings_fp32.npy"
    np.save(emb_path, embeddings)
    print(f"Saved embeddings → {emb_path}")

    # ---- Save metadata ----
    # Keep only lightweight columns needed for retrieval / display
    meta_cols = [c for c in ["parent_asin", "title", "price", "average_rating",
                              "rating_number", "store", "image_url", "categories"]
                 if c in df.columns]
    meta_df = df[meta_cols].reset_index(drop=True)
    meta_path = output_dir / f"{stem}_metadata.parquet"
    meta_df.to_parquet(meta_path, index=False)
    print(f"Saved metadata  → {meta_path}")

    # ---- Summary ----
    size_mb = emb_path.stat().st_size / 1024 / 1024
    print(f"\n{'='*50}")
    print(f"  Vectors : {embeddings.shape[0]:,} × {embeddings.shape[1]}")
    print(f"  File    : {size_mb:.1f} MB  (FP32)")
    print(f"  Speed   : {n/elapsed:.0f} texts/s on {device}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
