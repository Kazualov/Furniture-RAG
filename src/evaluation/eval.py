import json
import math
import time
import requests
from src.evaluation.metrics import ndcg, recall_at_k, reciprocal_rank

API_URL = "http://localhost:8000/search"
# URL для сбора сырых метрик из Qdrant
QDRANT_METRICS_URL = "http://localhost:6333/metrics"
TOP_K = 10


def search(query, alpha=0.5, limit=10, exact=False):
    payload = {
        "query": query,
        "alpha": alpha,
        "limit": limit,
        "exact": exact
    }

    # Замеряем время начала запроса
    t0 = time.perf_counter()
    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    # Вычисляем латентность в миллисекундах
    latency_ms = (time.perf_counter() - t0) * 1000.0

    data = response.json()
    product_ids = [item["product_id"] for item in data]

    return product_ids, latency_ms


def get_qdrant_ram_mb():
    """Пытается прочитать реальное потребление памяти Qdrant через Prometheus-метрики."""
    try:
        response = requests.get(QDRANT_METRICS_URL, timeout=2)
        if response.status_code == 200:
            for line in response.text.splitlines():
                if not line.startswith("#") and "memory" in line and "resident" in line:
                    return float(line.split()[-1]) / (1024 * 1024)
    except Exception:
        pass
    return None


def evaluate(golden_dataset, is_exact=True):
    recall_scores = []
    mrr_scores = []
    ndcg_scores = []
    latencies = []

    approach_name = "v1.0 Baseline (Flat/Exact)" if is_exact else "v1.1 Optimized (HNSW)"
    print(f"\nЗапуск оценки для подхода: {approach_name}")
    print(f"Обработка {len(golden_dataset)} запросов...")

    print(f"Прогрев системы для {approach_name}...")
    # Делаем холостой запрос, результаты и время которого мы просто игнорируем
    _ = search("warmup query", exact=is_exact, limit=1)

    for sample in golden_dataset:
        # Теперь функция search возвращает и результаты, и время выполнения
        prediction, latency = search(sample["query"], exact=is_exact, limit=TOP_K)

        relevant = sample["relevant_product_ids"]

        recall_scores.append(recall_at_k(relevant, prediction, TOP_K))
        mrr_scores.append(reciprocal_rank(relevant, prediction))
        ndcg_scores.append(ndcg(relevant, prediction, TOP_K))
        latencies.append(latency)

    # Запрашиваем RAM в конце прогона
    qdrant_ram = get_qdrant_ram_mb()

    print(f"\n=== РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ({approach_name}) ===")
    print(f"Качество поиска:")
    print(f"  Recall@{TOP_K}: {sum(recall_scores) / len(recall_scores):.4f}")
    print(f"  MRR:           {sum(mrr_scores) / len(mrr_scores):.4f}")
    print(f"  NDCG@{TOP_K}:  {sum(ndcg_scores) / len(ndcg_scores):.4f}")
    print(f"Производительность системы:")
    print(f"  Avg Latency:   {sum(latencies) / len(latencies):.2f} мс")
    if qdrant_ram:
        print(f"  Qdrant RAM:    {qdrant_ram:.1f} MB")
    else:
        print(f"  Qdrant RAM:    Н/Д (Снимите через `docker stats`)")
    print(f"{'=' * 40}\n")


if __name__ == "__main__":
    # Загружаем золотой датасет
    with open("src/evaluation/evaluation_dataset.json", encoding="utf-8") as f:
        golden_dataset = json.load(f)

    # 1. Сначала тестируем Baseline (Игнорируя HNSW граф)
    evaluate(golden_dataset, is_exact=True)

    # 2. Затем тестируем быструю версию с HNSW
    evaluate(golden_dataset, is_exact=False)
