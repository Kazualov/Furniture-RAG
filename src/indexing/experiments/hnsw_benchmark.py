"""
Эксперименты с параметрами графа HNSW в Qdrant.

Для сетки (m, ef_construct) меряет время построения индекса, потребление
памяти, латентность и recall@k относительно точного (Flat) поиска. Результаты
сохраняются в experiments/results/ (CSV + markdown).

    python -m src.indexing.experiments.hnsw_benchmark --vectors ./data/vectors.parquet
    python -m src.indexing.experiments.hnsw_benchmark \
        --vectors ./data/vectors.npy \
        --m 8,16,32 --ef-construct 100,200 --queries 200 --k 10
"""

import argparse
import json
import pathlib
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    SearchParams,
    VectorParams,
)

from src.indexing import config

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
BENCH_COLLECTION = "bench_hnsw"
UPLOAD_BATCH = 512


# --------------------------------------------------------------------------- #
# Загрузка векторов
# --------------------------------------------------------------------------- #
def load_vectors(path: str, limit: int | None = None) -> np.ndarray:
    """Грузит матрицу векторов (N, dim) из .npy или .parquet (колонка 'vector').

    limit — взять только первые N векторов (для быстрых прогонов на подвыборке).
    """
    p = pathlib.Path(path)
    if p.suffix == ".npy":
        arr = np.load(p, mmap_mode="r")          # mmap: не тянем весь файл в RAM сразу
        arr = arr[:limit] if limit else arr
        vecs = np.ascontiguousarray(arr, dtype=np.float32)
    elif p.suffix == ".parquet":
        df = pd.read_parquet(p)
        if "vector" not in df.columns:
            raise ValueError("В parquet ожидается колонка 'vector'.")
        if limit:
            df = df.iloc[:limit]
        vecs = np.ascontiguousarray(np.vstack(df["vector"].to_numpy()), dtype=np.float32)
    else:
        raise ValueError(f"Неподдерживаемый формат: {p.suffix} (нужен .npy или .parquet)")

    if vecs.shape[1] != config.EMBEDDING_DIM:
        raise ValueError(
            f"Размерность {vecs.shape[1]} != EMBEDDING_DIM={config.EMBEDDING_DIM}."
        )
    return vecs


# --------------------------------------------------------------------------- #
# Замеры
# --------------------------------------------------------------------------- #
def recreate_collection(client: QdrantClient, m: int, ef_construct: int) -> None:
    """Пересоздаёт bench-коллекцию с заданным HNSW. m=0 -> Flat (без графа).

    ef_construct зажимается до минимума 4 (требование Qdrant даже при m=0).
    """
    ef_construct = max(ef_construct, 4)
    if BENCH_COLLECTION in [c.name for c in client.get_collections().collections]:
        client.delete_collection(BENCH_COLLECTION)
    client.create_collection(
        collection_name=BENCH_COLLECTION,
        vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=m, ef_construct=ef_construct),
        # низкий порог индексации -> индекс строится даже на малых датасетах
        optimizers_config=OptimizersConfigDiff(indexing_threshold=1, default_segment_number=2),
    )


def upload_and_wait(client: QdrantClient, vecs: np.ndarray, timeout_s: int = 600) -> float:
    """Заливает точки и ждёт окончания индексации. Возвращает время построения (с)."""
    n = len(vecs)
    t0 = time.perf_counter()
    for i in range(0, n, UPLOAD_BATCH):
        chunk = vecs[i : i + UPLOAD_BATCH]
        points = [
            PointStruct(id=i + j, vector=chunk[j].tolist(), payload={"row": i + j})
            for j in range(len(chunk))
        ]
        client.upsert(collection_name=BENCH_COLLECTION, points=points, wait=True)

    # ждём, пока коллекция станет 'green' и проиндексирует все вектора
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        info = client.get_collection(BENCH_COLLECTION)
        indexed = info.indexed_vectors_count or 0
        if str(info.status).lower().endswith("green") and (indexed >= n or indexed == 0):
            break
        time.sleep(0.5)
    return time.perf_counter() - t0


def measure_search(client: QdrantClient, queries: np.ndarray, k: int,
                   exact: bool, hnsw_ef: int | None = None) -> tuple[list[list[int]], float]:
    """Прогоняет запросы. Возвращает (списки id top-k, средняя латентность мс)."""
    params = SearchParams(exact=exact)
    if not exact and hnsw_ef is not None:
        params = SearchParams(hnsw_ef=hnsw_ef, exact=False)

    all_ids: list[list[int]] = []
    latencies: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        resp = client.query_points(
            collection_name=BENCH_COLLECTION,
            query=q.tolist(),
            limit=k,
            search_params=params,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)
        all_ids.append([p.id for p in resp.points])
    return all_ids, float(np.mean(latencies))


def recall_at_k(approx: list[list[int]], truth: list[list[int]]) -> float:
    """Средний recall@k: доля истинных соседей, найденных приближённым поиском."""
    scores = []
    for a, t in zip(approx, truth):
        if not t:
            continue
        scores.append(len(set(a) & set(t)) / len(t))
    return float(np.mean(scores)) if scores else 0.0


def estimate_memory_mb(n: int, dim: int, m: int) -> float:
    """Грубая оценка памяти индекса: вектора (fp32) + рёбра графа HNSW."""
    vectors_bytes = n * dim * 4
    # ~ n * m * 2 связей * 4 байта (id). m=0 -> только вектора (Flat).
    graph_bytes = n * m * 2 * 4
    return (vectors_bytes + graph_bytes) / (1024 * 1024)


def read_qdrant_metrics_mb() -> float | None:
    """Best-effort чтение RSS Qdrant из /metrics (Prometheus). None, если недоступно."""
    url = f"http://{config.QDRANT_HOST}:{config.QDRANT_HTTP_PORT}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, OSError):
        return None
    for line in text.splitlines():
        # имя метрики зависит от версии Qdrant; ищем что-то про resident memory
        if line.startswith("#"):
            continue
        if "memory" in line and "resident" in line:
            try:
                return float(line.split()[-1]) / (1024 * 1024)
            except (ValueError, IndexError):
                continue
    return None


# --------------------------------------------------------------------------- #
# Главный цикл
# --------------------------------------------------------------------------- #
def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Бенчмарк HNSW в Qdrant")
    p.add_argument("--vectors", default=config.VECTORS_PATH, help="путь к .npy или .parquet")
    p.add_argument("--m", type=parse_int_list, default=[8, 16, 32])
    p.add_argument("--ef-construct", type=parse_int_list, default=[100, 200])
    p.add_argument("--search-ef", type=int, default=128, help="hnsw_ef на время поиска")
    p.add_argument("--queries", type=int, default=200, help="число тестовых запросов")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--limit", type=int, default=None,
                   help="взять только первые N векторов (быстрый прогон на подвыборке)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Гружу вектора из {args.vectors} ...")
    vecs = load_vectors(args.vectors, limit=args.limit)
    n, dim = vecs.shape
    print(f"Загружено {n} векторов размерности {dim}"
          + (f" (подвыборка --limit {args.limit})" if args.limit else ""))

    rng = np.random.default_rng(args.seed)
    q_idx = rng.choice(n, size=min(args.queries, n), replace=False)
    queries = vecs[q_idx]

    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_HTTP_PORT)

    rows = []

    # --- Baseline Flat (точный поиск, ground truth для recall) ---
    print("\n=== Flat baseline (exact=True) ===")
    recreate_collection(client, m=0, ef_construct=4)  # m=0 -> Flat; ef_construct тут не влияет
    build_flat = upload_and_wait(client, vecs)
    truth_ids, flat_latency = measure_search(client, queries, args.k, exact=True)
    rows.append({
        "config": "Flat (exact)",
        "m": 0,
        "ef_construct": 0,
        "build_time_s": round(build_flat, 2),
        "est_memory_mb": round(estimate_memory_mb(n, dim, 0), 1),
        "qdrant_memory_mb": read_qdrant_metrics_mb(),
        "avg_latency_ms": round(flat_latency, 3),
        "recall@k": 1.0,
    })
    print(f"build={build_flat:.2f}s  latency={flat_latency:.3f}ms  (recall=1.0 by definition)")

    # --- Сетка HNSW ---
    for m in args.m:
        for ef_c in args.ef_construct:
            print(f"\n=== HNSW m={m}, ef_construct={ef_c} ===")
            recreate_collection(client, m=m, ef_construct=ef_c)
            build_t = upload_and_wait(client, vecs)
            approx_ids, latency = measure_search(
                client, queries, args.k, exact=False, hnsw_ef=args.search_ef
            )
            rec = recall_at_k(approx_ids, truth_ids)
            mem_est = estimate_memory_mb(n, dim, m)
            rows.append({
                "config": f"HNSW m={m}, ef_c={ef_c}",
                "m": m,
                "ef_construct": ef_c,
                "build_time_s": round(build_t, 2),
                "est_memory_mb": round(mem_est, 1),
                "qdrant_memory_mb": read_qdrant_metrics_mb(),
                "avg_latency_ms": round(latency, 3),
                "recall@k": round(rec, 4),
            })
            print(f"build={build_t:.2f}s  latency={latency:.3f}ms  "
                  f"recall@{args.k}={rec:.4f}  est_mem={mem_est:.1f}MB")

    # cleanup bench-коллекции
    client.delete_collection(BENCH_COLLECTION)

    # --- Сохранение результатов ---
    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "hnsw_benchmark.csv"
    md_path = RESULTS_DIR / "hnsw_benchmark.md"
    df.to_csv(csv_path, index=False)

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# HNSW benchmark (N={n}, dim={dim}, k={args.k}, "
                f"queries={len(queries)}, search_ef={args.search_ef})\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\nПримечание: `qdrant_memory_mb` берётся из /metrics (если доступно), "
                "иначе пусто — точный RSS можно снять командой "
                "`docker stats furniture_qdrant`. `est_memory_mb` — аналитическая оценка "
                "(вектора fp32 + рёбра графа).\n")

    print(f"\nГотово. Результаты:\n  {csv_path}\n  {md_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
