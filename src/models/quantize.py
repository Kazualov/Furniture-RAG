"""
Stage 2: Quantization & Speed Benchmark
========================================
Applies INT8 static/dynamic quantization to all-MiniLM-L6-v2,
benchmarks FP32 vs INT8 on a sample, and saves the optimized vectors.

Usage:
    python stage2_quantize.py --input office_products_micro.parquet
    python stage2_quantize.py --input office_products_micro.parquet --quant-type static
"""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.quantization
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BENCHMARK_SAMPLE = 500   # rows used for speed comparison
DEFAULT_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def apply_dynamic_quantization(model: SentenceTransformer) -> SentenceTransformer:
    """
    Dynamic INT8: weights quantized at load time, activations quantized at
    runtime per-batch. No calibration data needed. Fast and easy.
    """
    quantized = copy.deepcopy(model)
    torch.quantization.quantize_dynamic(
        quantized[0].auto_model,          # the underlying HF transformer
        {torch.nn.Linear},                 # quantize Linear layers only
        dtype=torch.qint8,
        inplace=True,
    )
    return quantized


def apply_static_quantization(
    model: SentenceTransformer,
    calibration_texts: list[str],
    batch_size: int = 32,
) -> SentenceTransformer:
    """
    Static INT8: both weights and activations are quantized using scale/zero-point
    computed from calibration data. Potentially faster than dynamic on CPU,
    but requires a representative calibration set.

    Note: static quantization on transformer encoder blocks can be unstable
    with some PyTorch versions; dynamic is the safer default.
    """
    quantized = copy.deepcopy(model)
    transformer_model = quantized[0].auto_model

    transformer_model.eval()

    qconfig = torch.quantization.get_default_qconfig("x86")
    transformer_model.qconfig = qconfig

    torch.quantization.prepare(transformer_model, inplace=True)

    # Calibration pass (используем встроенный батчинг вместо ручного цикла)
    print("  Running calibration pass...")
    with torch.no_grad():
        quantized.encode(
            calibration_texts, 
            batch_size=batch_size, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )

    torch.quantization.convert(transformer_model, inplace=True)
    return quantized


# ---------------------------------------------------------------------------
# Encoding helper
# ---------------------------------------------------------------------------

def encode_timed(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    label: str,
) -> tuple[np.ndarray, float]:
    """Encode texts using built-in batching, return (embeddings, seconds_elapsed)."""
    print(f" Running encoding: {label}")
    t0 = time.perf_counter()
    
    # Библиотека сама эффективно разбивает на батчи и выводит прогресс-бар
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    elapsed = time.perf_counter() - t0
    return embeddings.astype(np.float32), elapsed


# ---------------------------------------------------------------------------
# Benchmark report
# ---------------------------------------------------------------------------

def print_benchmark(
    n: int,
    fp32_time: float,
    int8_time: float,
    fp32_emb: np.ndarray,
    int8_emb: np.ndarray,
    quant_type: str,
):
    speedup = fp32_time / int8_time if int8_time > 0 else float("inf")

    # Cosine similarity between FP32 and INT8 vectors (mean over sample)
    cos_sim = float(np.mean(np.sum(fp32_emb * int8_emb, axis=1)))  # both L2-normed

    print(f"\n{'='*55}")
    print(f"  Benchmark results  ({quant_type} INT8 vs FP32, n={n})")
    print(f"{'='*55}")
    print(f"  FP32   : {fp32_time:.2f}s  →  {n/fp32_time:.0f} texts/s")
    print(f"  INT8   : {int8_time:.2f}s  →  {n/int8_time:.0f} texts/s")
    print(f"  Speedup: {speedup:.2f}×")
    print(f"  Mean cosine similarity (FP32 vs INT8): {cos_sim:.4f}")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Quantization")
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
    )
    parser.add_argument(
        "--quant-type",
        choices=["dynamic", "static"],
        default="dynamic",
        help="Quantization strategy (dynamic is recommended)",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Device (quantization works on CPU only) ----
    device = "cpu"
    print(f"Note: PyTorch quantization runs on CPU only.")

    # ---- Load data ----
    df = pd.read_parquet(args.input)
    # Предполагается, что на шаге 1 была подготовлена колонка 'full_text'
    texts = df["full_text"].fillna("").tolist()
    texts = [t if t.strip() else " " for t in texts]
    print(f"Total texts: {len(texts):,}")

    benchmark_texts = texts[: BENCHMARK_SAMPLE]
    all_texts = texts  # full dataset for final INT8 run

    # ---- Load FP32 model ----
    print(f"\nLoading FP32 model: {args.model}")
    fp32_model = SentenceTransformer(args.model, device=device)

    # ---- Quantize ----
    print(f"\nApplying {args.quant_type} INT8 quantization...")
    if args.quant_type == "static":
        int8_model = apply_static_quantization(
            fp32_model, benchmark_texts, batch_size=args.batch_size
        )
    else:
        int8_model = apply_dynamic_quantization(fp32_model)

    # ---- Benchmark on sample ----
    print(f"\nBenchmark on {BENCHMARK_SAMPLE} texts...")
    fp32_sample, fp32_time = encode_timed(
        fp32_model, benchmark_texts, args.batch_size, "FP32 Sample"
    )
    int8_sample, int8_time = encode_timed(
        int8_model, benchmark_texts, args.batch_size, "INT8 Sample"
    )

    print_benchmark(
        BENCHMARK_SAMPLE,
        fp32_time,
        int8_time,
        fp32_sample,
        int8_sample,
        args.quant_type,
    )

    # ---- Encode full dataset with INT8 ----
    print("Encoding full dataset with INT8 model...")
    int8_full, total_elapsed = encode_timed(
        int8_model, all_texts, args.batch_size, "INT8 Full Dataset"
    )

    # ---- Save INT8 vectors (Квантование самих эмбеддингов для диска) ----
    stem = Path(args.input).stem
    int8_path = output_dir / f"{stem}_embeddings_int8_{args.quant_type}.npy"
    scale_path = output_dir / f"{stem}_embeddings_int8_{args.quant_type}_scale.npy"

    # Векторы L2-нормированы, но при 384 измерениях типичная компонента
    # имеет величину ~1/sqrt(384) ≈ 0.05, а не ~1.0 — фиксированный масштаб
    # *127 использовал бы лишь малую часть диапазона INT8 (низкая точность).
    # Вместо этого считаем масштаб отдельно для КАЖДОГО вектора по его
    # реальному max-abs, чтобы использовать весь диапазон [-128, 127].
    abs_max = np.max(np.abs(int8_full), axis=1, keepdims=True)      # (N, 1)
    abs_max = np.where(abs_max == 0, 1.0, abs_max)                   # защита от деления на 0
    scale = 127.0 / abs_max                                          # (N, 1)

    # np.round (не .astype напрямую!) — .astype(np.int8) усекает дробную
    # часть в сторону нуля, что вносит систематическое смещение (bias).
    int8_vectors = np.round(int8_full * scale).clip(-128, 127).astype(np.int8)

    np.save(int8_path, int8_vectors)
    np.save(scale_path, scale.astype(np.float32).squeeze())

    # Расчет размеров для красивого вывода в консоль
    size_fp32 = (int8_full.shape[0] * int8_full.shape[1] * 4) / 1024 / 1024  # float32 = 4 байта
    size_int8_file = int8_path.stat().st_size / 1024 / 1024                 # int8 = 1 байт

    print(f"Saved INT8 embeddings → {int8_path}")
    print(f"Saved scale factors   → {scale_path}  (обязательны для dequantize)")
    print(f"\n{'='*55}")
    print(f"  Stage 2 complete")
    print(f"  Shape          : {int8_full.shape}")
    print(f"  Theoretical FP32 size : {size_fp32:.1f} MB")
    print(f"  Saved file size       : {size_int8_file:.1f} MB")
    print(f"  Full-dataset speed    : {len(all_texts)/total_elapsed:.0f} texts/s")
    print(f"{'='*55}")

    # ---- Save benchmark report as CSV ----
    report_path = output_dir / f"{stem}_benchmark_{args.quant_type}.csv"
    pd.DataFrame([{
        "quant_type": args.quant_type,
        "n_benchmark": BENCHMARK_SAMPLE,
        "fp32_time_s": round(fp32_time, 3),
        "int8_time_s": round(int8_time, 3),
        "speedup_x": round(fp32_time / int8_time, 3),
        "mean_cosine_sim": round(float(np.mean(np.sum(fp32_sample * int8_sample, axis=1))), 4),
        "total_texts": len(all_texts),
    }]).to_csv(report_path, index=False)
    print(f"Benchmark report  → {report_path}")


if __name__ == "__main__":
    main()
