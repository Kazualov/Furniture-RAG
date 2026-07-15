import json
import math
import time
import requests
from src.evaluation.metrics import ndcg, recall_at_k, reciprocal_rank

API_URL = "http://localhost:8000/search"
QDRANT_METRICS_URL = "http://localhost:6333/metrics"

# Увеличили глубину поиска до 50
TOP_K = 10


def search(query, alpha=0.55, limit=TOP_K, exact=False):
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
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / (1024 * 1024)
    except Exception:
        pass
    return None


def precision_at_k(relevant, prediction, k):
    """Считает Precision@K: доля релевантных среди выданных K."""
    if not prediction:
        return 0.0
    # Берем только топ-K предсказаний
    pred_k = prediction[:k]
    # Считаем пересечение с релевантными
    hits = len(set(relevant).intersection(set(pred_k)))
    return hits / len(pred_k)


if __name__ == "__main__":
    # Предполагаем, что evaluation_dataset.json лежит там же, где и раньше
    with open("src/evaluation/evaluation_dataset_2.json", "r", encoding="utf-8") as f:
        evaluation_dataset = json.load(f)

    approaches = [
        {"name": "v1.0 Baseline (Flat/Exact)", "is_exact": True},
        {"name": "v1.1 Optimized (HNSW)", "is_exact": False}
    ]

    # === ПРОГРЕВ СИСТЕМЫ ===
    # Делаем пустой запрос, чтобы загрузить модель в RAM и прогреть кэши БД
    print("Прогрев системы перед тестированием...")
    try:
        _ = search("warmup", limit=1)
    except requests.exceptions.RequestException as e:
        print(f"Внимание: Не удалось выполнить прогревочный запрос. Проверь, запущен ли API. Ошибка: {e}")

    for approach in approaches:
        approach_name = approach["name"]
        is_exact = approach["is_exact"]

        print(f"\nЗапуск оценки для подхода: {approach_name}")
        print(f"Обработка {len(evaluation_dataset)} запросов...")

        recall_scores = []
        mrr_scores = []
        ndcg_scores = []
        precision_scores = []
        latencies = []

        for sample in evaluation_dataset:
            prediction, latency = search(sample["query"], exact=is_exact, limit=TOP_K)

            relevant = sample["relevant_product_ids"]

            recall_scores.append(recall_at_k(relevant, prediction, TOP_K))
            mrr_scores.append(reciprocal_rank(relevant, prediction))
            ndcg_scores.append(ndcg(relevant, prediction, TOP_K))
            precision_scores.append(precision_at_k(relevant, prediction, TOP_K))
            latencies.append(latency)

        # Запрашиваем RAM в конце прогона
        qdrant_ram = get_qdrant_ram_mb()

        print(f"\n=== РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ({approach_name}) ===")
        print(f"Качество поиска (Top-{TOP_K}):")
        print(f"  Recall@{TOP_K}:    {sum(recall_scores) / len(recall_scores):.4f}")
        print(f"  Precision@{TOP_K}: {sum(precision_scores) / len(precision_scores):.4f}")
        print(f"  MRR:           {sum(mrr_scores) / len(mrr_scores):.4f}")
        print(f"  NDCG@{TOP_K}:     {sum(ndcg_scores) / len(ndcg_scores):.4f}")
        print(f"Производительность системы:")
        print(f"  Avg Latency:   {sum(latencies) / len(latencies):.2f} мс")
        if qdrant_ram:
            print(f"  Qdrant RAM:    {qdrant_ram:.1f} MB")
        else:
            print(f"  Qdrant RAM:    Н/Д (Сбор метрик недоступен)")
        print("========================================")
