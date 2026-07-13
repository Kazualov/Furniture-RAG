import argparse
import time
from pathlib import Path

import cv2  # Используем OpenCV вместо PIL
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Используем мультимодальный CLIP
MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
DEFAULT_BATCH_SIZE = 64
EMBED_DIM = 512  # Выходная размерность для clip-ViT-B-32


def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows from {path}")
    print(f"Columns: {list(df.columns)}")
    return df


def encode_multimodal_in_batches(
    model: SentenceTransformer,
    df: pd.DataFrame,
    image_dir: Path,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """
    Побатчево кодирует товары. Если находит локальное изображение товара,
    кодирует его визуальный эмбеддинг через OpenCV. Если картинки нет — делает
    фоллбек на текстовое описание, проецируя его в то же пространство.
    """
    all_embeddings = []

    for start in tqdm(range(0, len(df), batch_size), desc="Encoding multimodal batches"):
        batch_df = df.iloc[start : start + batch_size]
        batch_inputs = []

        for _, row in batch_df.iterrows():
            # Предположим, имя файла картинки — это parent_asin.jpg
            img_path = image_dir / f"{row['parent_asin']}.jpg"
            img_loaded = False

            if img_path.exists():
                try:
                    # Читаем изображение через OpenCV
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # Конвертируем BGR -> RGB (критично для CLIP!)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        batch_inputs.append(img_rgb)
                        img_loaded = True
                except Exception as e:
                    print(f"Warning: Failed to load image {img_path}: {e}")

            if not img_loaded:
                # Фоллбек на текст, если картинки нет или она повреждена
                fallback_text = row.get("full_text", row.get("title", " "))
                batch_inputs.append(str(fallback_text if fallback_text.strip() else " "))

        with torch.no_grad():
            embs = model.encode(
                batch_inputs,
                batch_size=len(batch_inputs),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Косинусное сходство в Qdrant
                device=device,
            )
        all_embeddings.append(embs)

    return np.vstack(all_embeddings).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate Multimodal Embeddings using OpenCV")
    parser.add_argument(
        "--input",
        default="embeddings/office_products_micro.parquet",
        help="Path to input .parquet file",
    )
    parser.add_argument(
        "--image-dir",
        default="data/images",
        help="Directory where product images are stored",
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
        help="Encoding batch size",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df = load_parquet(args.input)

    print(f"Loading multimodal model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    t0 = time.perf_counter()
    embeddings = encode_multimodal_in_batches(model, df, image_dir, args.batch_size, device)
    elapsed = time.perf_counter() - t0

    n = len(df)
    print(f"\nEncoding done: {n:,} items in {elapsed:.1f}s")

    # Сохраняем эмбеддинги
    stem = Path(args.input).stem
    emb_path = output_dir / f"{stem}_embeddings_fp32.npy"
    np.save(emb_path, embeddings)

    # Сохраняем метаданные (ваша исходная логика)
    meta_cols = [c for c in ["parent_asin", "title", "price", "average_rating",
                              "rating_number", "store", "image_url", "categories"] if c in df.columns]
    meta_df = df[meta_cols].reset_index(drop=True)
    meta_df.to_parquet(output_dir / f"{stem}_metadata.parquet", index=False)
    print(f"Saved metadata & embeddings to {output_dir}")


if __name__ == "__main__":
    main()