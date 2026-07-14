import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.quantization
from sentence_transformers import SentenceTransformer

if torch.backends.quantized.engine == 'none':
    torch.backends.quantized.engine = 'qnnpack'  # macOS / ARM fallback


def get_native_qengine() -> str:
    """
    Выбирает нативный движок квантизации под текущую платформу и
    устанавливает его как активный.

    - qnnpack   — ARM (Apple Silicon, мобильные устройства)
    - x86       — Intel/AMD, актуальные версии PyTorch (>= 1.12)
    - fbgemm    — Intel/AMD, старые версии PyTorch (fallback)

    ВАЖНО: значение, установленное здесь, всегда должно совпадать с
    движком, который передаётся в get_default_qconfig(...) — иначе
    PyTorch падает с NotImplementedError при вызове quantized-слоя.
    """
    supported = torch.backends.quantized.supported_engines
    current = torch.backends.quantized.engine

    if current in supported and current != 'none':
        # Движок уже корректно выставлен (например, кто-то задал вручную) — используем его
        return current

    if 'qnnpack' in supported and current in ('none', 'qnnpack'):
        preferred = 'qnnpack'
    elif 'x86' in supported:
        preferred = 'x86'
    elif 'fbgemm' in supported:
        preferred = 'fbgemm'
    else:
        preferred = supported[0]

    torch.backends.quantized.engine = preferred
    return preferred

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BENCHMARK_SAMPLE = 500
DEFAULT_BATCH_SIZE = 64


def apply_dynamic_quantization(model: SentenceTransformer) -> SentenceTransformer:
    """
    Dynamic INT8 quantization с корректной настройкой движка под платформу.
    """
    quantized = copy.deepcopy(model)

    get_native_qengine()  # выставляет правильный движок под текущую платформу

    # Применяем квантование
    torch.quantization.quantize_dynamic(
        quantized[0].auto_model,
        {torch.nn.Linear},
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
    Static INT8 quantization.
    """
    # Устанавливаем движок и используем ТУ ЖЕ переменную для qconfig —
    # иначе global engine и qconfig могут разойтись (например, engine='x86'
    # на Linux, но qconfig захардкожен на 'qnnpack') и PyTorch упадёт с
    # NotImplementedError при первом вызове квантованного слоя.
    engine = get_native_qengine()

    quantized = copy.deepcopy(model)
    transformer_model = quantized[0].auto_model
    transformer_model.eval()

    # Используем современную конфигурацию — движок совпадает с тем, что
    # реально активен в torch.backends.quantized.engine
    qconfig = torch.quantization.get_default_qconfig(engine)
    transformer_model.qconfig = qconfig

    torch.quantization.prepare(transformer_model, inplace=True)

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


def encode_timed(
        model: SentenceTransformer,
        texts: list[str],
        batch_size: int,
        label: str,
) -> tuple[np.ndarray, float]:
    """Encode texts and return embeddings with timing."""
    print(f" Running encoding: {label}")
    t0 = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
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
    print(f"  Benchmark results  ({quant_type} INT8 vs FP32, n={n})")
    print(f"{'=' * 55}")
    print(f"  FP32   : {fp32_time:.2f}s  →  {n / fp32_time:.0f} texts/s")
    print(f"  INT8   : {int8_time:.2f}s  →  {n / int8_time:.0f} texts/s")
    print(f"  Speedup: {speedup:.2f}×")
    print(f"  Mean cosine similarity: {cos_sim:.4f}")
    print(f"{'=' * 55}\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Quantization")
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
        choices=["dynamic", "static"],
        default="dynamic",
        help="Quantization strategy",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Note: PyTorch quantization runs on CPU only.")
    print(f"Using quantized engine: {torch.backends.quantized.engine}")

    if args.quant_type == "static":
        torch_major_minor = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        if torch_major_minor >= (2, 10):
            print(
                "\n⚠️  Warning: torch >= 2.10 removes the legacy eager-mode "
                "torch.quantization API used by static quantization here.\n"
                "   If this fails with a quantized-op error, either pin an "
                "older torch (torch>=2.2.0,<2.10) or use --quant-type dynamic instead.\n"
            )

    # Load data
    df = pd.read_parquet(args.input)
    texts = df["full_text"].fillna("").tolist()
    texts = [t if t.strip() else " " for t in texts]
    print(f"Total texts: {len(texts):,}")

    benchmark_texts = texts[:BENCHMARK_SAMPLE]
    all_texts = texts

    # Load FP32 model
    print(f"\nLoading FP32 model: {args.model}")
    fp32_model = SentenceTransformer(args.model, device="cpu")

    # Quantize
    print(f"\nApplying {args.quant_type} INT8 quantization...")
    if args.quant_type == "static":
        int8_model = apply_static_quantization(
            fp32_model, benchmark_texts, batch_size=args.batch_size
        )
    else:
        int8_model = apply_dynamic_quantization(fp32_model)

    # Benchmark
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

    # Encode full dataset
    print("Encoding full dataset with INT8 model...")
    int8_full, total_elapsed = encode_timed(
        int8_model, all_texts, args.batch_size, "INT8 Full Dataset"
    )

    # Save results
    stem = Path(args.input).stem
    int8_path = output_dir / f"{stem}_embeddings_int8_{args.quant_type}.npy"
    scale_path = output_dir / f"{stem}_embeddings_int8_{args.quant_type}_scale.npy"

    # Quantize embeddings for storage
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
