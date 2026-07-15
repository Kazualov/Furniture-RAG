import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Импортируем TorchAO
from torchao.quantization import (
    quantize_,
    int8_dynamic_activation_int8_weight,
    int8_weight_only
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BENCHMARK_SAMPLE = 500
DEFAULT_BATCH_SIZE = 64


def apply_dynamic_quantization(model: SentenceTransformer) -> SentenceTransformer:
    """
    Dynamic INT8 (W8A8) quantization с помощью TorchAO.
    Квантует веса в INT8 и динамически квантует активации при inference.
    """
    quantized = copy.deepcopy(model)

    # TorchAO применяет квантование in-place.
    # Мы применяем его к underlying transformer модели (auto_model).
    quantize_(quantized[0].auto_model, int8_dynamic_activation_int8_weight())

    return quantized


def apply_weight_only_quantization(model: SentenceTransformer) -> SentenceTransformer:
    """
    Weight-only INT8 (W8A16/W8A32) quantization с помощью TorchAO.
    Заменяет "static", так как на GPU часто выгоднее квантовать только веса,
    избегая накладных расходов на квантование/деквантование активаций.
    """
    quantized = copy.deepcopy(model)

    quantize_(quantized[0].auto_model, int8_weight_only())

    return quantized


def encode_timed(
        model: SentenceTransformer,
        texts: list[str],
        batch_size: int,
        label: str,
) -> tuple[np.ndarray, float]:
    """Encode texts and return embeddings with timing."""
    print(f" Running encoding: {label}")

    # Делаем warmup (особенно важно для скомпилированных моделей на GPU)
    _ = model.encode(texts[:2], show_progress_bar=False)

    t0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    # Синхронизируем CUDA перед остановкой таймера
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    return embeddings.astype(np.float32), elapsed


def print_benchmark(
        n: int,
        fp32_time: float,
        int8_time: float,
        fp32_emb: np.ndarray,
        int8_emb: np.ndarray,
        quant_type: str,
):
    speedup = fp32_time / int8_time if int8_time > 0 else float("inf")
    cos_sim = float(np.mean(np.sum(fp32_emb * int8_emb, axis=1)))

    print(f"\n{'=' * 55}")
    print(f"  Benchmark results  ({quant_type.upper()} INT8 vs FP32, n={n})")
    print(f"{'=' * 55}")
    print(f"  FP32   : {fp32_time:.2f}s  →  {n / fp32_time:.0f} texts/s")
    print(f"  INT8   : {int8_time:.2f}s  →  {n / int8_time:.0f} texts/s")
    print(f"  Speedup: {speedup:.2f}×")
    print(f"  Mean cosine similarity: {cos_sim:.4f}")
    print(f"{'=' * 55}\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: GPU Quantization with TorchAO")
    parser.add_argument(
        "--input",
        required=True,
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
        choices=["dynamic", "weight_only"],
        default="dynamic",
        help="Quantization strategy for TorchAO",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Apply torch.compile (highly recommended for TorchAO speedup)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target device: {device.upper()}")
    if device == "cpu":
        print("⚠️ Warning: TorchAO is highly optimized for GPU. CPU performance may vary.")

    # Load data
    df = pd.read_parquet(args.input)
    texts = df["full_text"].fillna("").tolist()
    texts = [t if t.strip() else " " for t in texts]
    print(f"Total texts: {len(texts):,}")

    benchmark_texts = texts[:BENCHMARK_SAMPLE]
    all_texts = texts

    # Load FP32 (или FP16/BF16) model на GPU
    print(f"\nLoading base model: {args.model}")
    fp32_model = SentenceTransformer(args.model, device=device)

    # Для GPU бэйзлайна часто используют BF16 или FP16
    if device == "cuda":
        fp32_model[0].auto_model.to(torch.bfloat16)

    # Quantize
    print(f"\nApplying {args.quant_type} INT8 quantization via TorchAO...")
    if args.quant_type == "weight_only":
        int8_model = apply_weight_only_quantization(fp32_model)
    else:
        int8_model = apply_dynamic_quantization(fp32_model)

    # Компиляция графа (рекомендуется для TorchAO)
    if args.compile:
        print("Compiling model with torch.compile (this may take a few minutes)...")
        # Компилируем внутренний трансформер
        int8_model[0].auto_model = torch.compile(
            int8_model[0].auto_model,
            mode="max-autotune"
        )
        if hasattr(fp32_model[0], 'auto_model'):
            fp32_model[0].auto_model = torch.compile(
                fp32_model[0].auto_model,
                mode="max-autotune"
            )

    # Benchmark
    print(f"\nBenchmark on {BENCHMARK_SAMPLE} texts...")
    fp32_sample, fp32_time = encode_timed(
        fp32_model, benchmark_texts, args.batch_size, "Base (BF16/FP32) Sample"
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

    # Encode full dataset
    print("Encoding full dataset with INT8 model...")
    int8_full, total_elapsed = encode_timed(
        int8_model, all_texts, args.batch_size, "INT8 Full Dataset"
    )

    # Save results (выходные векторы квантуются как и раньше)
    stem = Path(args.input).stem
    int8_path = output_dir / f"{stem}_embeddings_int8_{args.quant_type}.npy"
    scale_path = output_dir / f"{stem}_embeddings_int8_{args.quant_type}_scale.npy"

    abs_max = np.max(np.abs(int8_full), axis=1, keepdims=True)
    abs_max = np.where(abs_max == 0, 1.0, abs_max)
    scale = 127.0 / abs_max
    int8_vectors = np.round(int8_full * scale).clip(-128, 127).astype(np.int8)

    np.save(int8_path, int8_vectors)
    np.save(scale_path, scale.astype(np.float32).squeeze())

    size_fp32 = (int8_full.shape[0] * int8_full.shape[1] * 4) / 1024 / 1024
    size_int8_file = int8_path.stat().st_size / 1024 / 1024

    print(f"Saved INT8 embeddings → {int8_path}")
    print(f"Saved scale factors   → {scale_path}")
    print(f"\n{'=' * 55}")
    print(f"  Stage 2 complete")
    print(f"  Shape          : {int8_full.shape}")
    print(f"  Theoretical FP32 size : {size_fp32:.1f} MB")
    print(f"  Saved file size       : {size_int8_file:.1f} MB")
    print(f"  Full-dataset speed    : {len(all_texts) / total_elapsed:.0f} texts/s")
    print(f"{'=' * 55}")

    # Save benchmark report
    report_path = output_dir / f"{stem}_benchmark_torchao_{args.quant_type}.csv"
    pd.DataFrame([{
        "quant_type": args.quant_type,
        "n_benchmark": BENCHMARK_SAMPLE,
        "base_time_s": round(fp32_time, 3),
        "int8_time_s": round(int8_time, 3),
        "speedup_x": round(fp32_time / int8_time, 3),
        "mean_cosine_sim": round(float(np.mean(np.sum(fp32_sample * int8_sample, axis=1))), 4),
        "total_texts": len(all_texts),
    }]).to_csv(report_path, index=False)
    print(f"Benchmark report  → {report_path}")


if __name__ == "__main__":
    main()