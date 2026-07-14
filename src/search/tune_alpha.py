"""
tune_alpha.py — подбор оптимального alpha для гибридного поиска (search/main.py)
по golden set с разметкой relevance.

Что делает:
  1. Загружает golden_set.csv (query_id, query, product_id, title, description, relevance)
  2. Для каждого alpha из сетки прогоняет каждый запрос через POST /search (TestClient)
  3. Считает NDCG@k и MRR, сравнивая выдачу с релевантностью из golden set
  4. Опционально сегментирует запросы на "keyword" (короткие, <=3 слов) и
     "semantic" (длиннее/описательные), чтобы показать trade-off на графике
  5. Сохраняет results.csv (метрики по каждой alpha) и alpha_sweep.png (график)
  6. Печатает рекомендованную alpha с обоснованием

Запуск:
    python tune_alpha.py --golden-set golden_set.csv --k 10
    python tune_alpha.py --golden-set golden_set.csv --alphas 0,0.1,0.2,...,1.0 --k 10 --bootstrap 500
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Поправьте импорт под структуру вашего проекта — как в test_api.py
from fastapi.testclient import TestClient
from src.search.main import app



# --------------------------------------------------------------------------
# Загрузка golden set
# --------------------------------------------------------------------------

def load_golden_set(path):
    """Возвращает dict: query_id -> {"query": str, "judgments": {product_id: relevance}}"""
    queries = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["query_id"]
            if qid not in queries:
                queries[qid] = {"query": row["query"], "judgments": {}}
            rel = row["relevance"].strip()
            queries[qid]["judgments"][row["product_id"]] = float(rel) if rel else 0.0
    return queries


def classify_query(query_text, keyword_max_words=3):
    """Грубая эвристика сегментации: короткие запросы = keyword, длинные/описательные = semantic."""
    return "keyword" if len(query_text.split()) <= keyword_max_words else "semantic"


# --------------------------------------------------------------------------
# Метрики ранжирования
# --------------------------------------------------------------------------

def dcg_at_k(relevances, k):
    return sum(
        (2 ** rel - 1) / math.log2(idx + 2)
        for idx, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(ranked_product_ids, judgments, k):
    gains = [judgments.get(pid, 0.0) for pid in ranked_product_ids]
    dcg = dcg_at_k(gains, k)
    ideal_gains = sorted(judgments.values(), reverse=True)
    idcg = dcg_at_k(ideal_gains, k)
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked_product_ids, judgments, relevance_threshold=1.0):
    for idx, pid in enumerate(ranked_product_ids):
        if judgments.get(pid, 0.0) >= relevance_threshold:
            return 1.0 / (idx + 1)
    return 0.0


# --------------------------------------------------------------------------
# Прогон поиска
# --------------------------------------------------------------------------

def search(client, query_text, alpha, limit):
    resp = client.post("/search", json={"query": query_text, "limit": limit, "alpha": alpha})
    resp.raise_for_status()
    return [item["product_id"] for item in resp.json()]

def evaluate_alpha(client, queries, alpha, k):
    """Возвращает per-query метрики: list of dicts {query_id, query, segment, ndcg, mrr}"""
    rows = []
    for qid, data in queries.items():
        # Передаем client внутрь search
        ranked = search(client, data["query"], alpha, limit=k)
        rows.append({
            "query_id": qid,
            "query": data["query"],
            "segment": classify_query(data["query"]),
            "ndcg": ndcg_at_k(ranked, data["judgments"], k),
            "mrr": mrr(ranked, data["judgments"]),
        })
    return rows


# --------------------------------------------------------------------------
# Bootstrap доверительный интервал (по запросам, с возвращением)
# --------------------------------------------------------------------------

def bootstrap_ci(values, n_iter=500, ci=0.95):
    if not values:
        return (0.0, 0.0)
    values = np.array(values)
    means = [np.mean(np.random.choice(values, size=len(values), replace=True)) for _ in range(n_iter)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return lo, hi


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-set", default="golden_set.csv")
    parser.add_argument("--alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--k", type=int, default=10, help="k для NDCG@k / глубина выдачи")
    parser.add_argument("--bootstrap", type=int, default=0, help="число bootstrap-итераций для CI (0 = выключено)")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    queries = load_golden_set(args.golden_set)
    print(f"Загружено {len(queries)} запросов из {args.golden_set}")

    all_rows = []
    summary = []

    # Инициируем TestClient как контекстный менеджер для корректной отработки lifespan (подключения к БД)
    with TestClient(app) as client:
        for alpha in alphas:
            print(f"Прогон alpha={alpha:.2f} ...")
            # Обязательно передаем объект client в evaluate_alpha
            rows = evaluate_alpha(client, queries, alpha, args.k)
            for r in rows:
                r["alpha"] = alpha
            all_rows.extend(rows)

            overall_ndcg = [r["ndcg"] for r in rows]
            overall_mrr = [r["mrr"] for r in rows]
            keyword_ndcg = [r["ndcg"] for r in rows if r["segment"] == "keyword"]
            semantic_ndcg = [r["ndcg"] for r in rows if r["segment"] == "semantic"]

            entry = {
                "alpha": alpha,
                "ndcg_mean": np.mean(overall_ndcg),
                "mrr_mean": np.mean(overall_mrr),
                "ndcg_keyword": np.mean(keyword_ndcg) if keyword_ndcg else None,
                "ndcg_semantic": np.mean(semantic_ndcg) if semantic_ndcg else None,
            }

            if args.bootstrap > 0:
                lo, hi = bootstrap_ci(overall_ndcg, n_iter=args.bootstrap)
                entry["ndcg_ci_lo"] = lo
                entry["ndcg_ci_hi"] = hi

            summary.append(entry)

    # --- Сохранить подробный csv (per query x alpha) ---
    detail_path = Path(args.out_dir) / "alpha_sweep_detail.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "query_id", "query", "segment", "ndcg", "mrr"])
        writer.writeheader()
        writer.writerows(all_rows)

    # --- Сохранить summary csv ---
    summary_path = Path(args.out_dir) / "alpha_sweep_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    # --- График ---
    fig, ax = plt.subplots(figsize=(8, 5))
    alphas_arr = [s["alpha"] for s in summary]
    ndcg_arr = [s["ndcg_mean"] for s in summary]
    ax.plot(alphas_arr, ndcg_arr, marker="o", label="NDCG@%d — overall" % args.k, linewidth=2)

    if all(s["ndcg_keyword"] is not None for s in summary):
        ax.plot(alphas_arr, [s["ndcg_keyword"] for s in summary], marker="s",
                 linestyle="--", label="NDCG@%d — keyword-запросы" % args.k, alpha=0.8)
    if all(s["ndcg_semantic"] is not None for s in summary):
        ax.plot(alphas_arr, [s["ndcg_semantic"] for s in summary], marker="^",
                 linestyle="--", label="NDCG@%d — semantic-запросы" % args.k, alpha=0.8)

    if args.bootstrap > 0:
        lo = [s["ndcg_ci_lo"] for s in summary]
        hi = [s["ndcg_ci_hi"] for s in summary]
        ax.fill_between(alphas_arr, lo, hi, alpha=0.15, color="C0", label="95% CI (overall)")

    best_alpha = max(summary, key=lambda s: s["ndcg_mean"])
    ax.axvline(best_alpha["alpha"], color="gray", linestyle=":", linewidth=1)
    ax.scatter([best_alpha["alpha"]], [best_alpha["ndcg_mean"]], color="red", zorder=5,
               label=f"лучшая alpha = {best_alpha['alpha']:.2f}")

    ax.set_xlabel("alpha (вес sparse-поиска)")
    ax.set_ylabel(f"NDCG@{args.k}")
    ax.set_title("Подбор alpha для гибридного поиска")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    chart_path = Path(args.out_dir) / "alpha_sweep.png"
    fig.savefig(chart_path, dpi=150)

    # --- Итог в консоль ---
    print("\n=== Результаты ===")
    print(f"{'alpha':>6} {'NDCG@%d' % args.k:>10} {'MRR':>8} {'kw NDCG':>10} {'sem NDCG':>10}")
    for s in summary:
        print(f"{s['alpha']:>6.2f} {s['ndcg_mean']:>10.4f} {s['mrr_mean']:>8.4f} "
              f"{(s['ndcg_keyword'] or 0):>10.4f} {(s['ndcg_semantic'] or 0):>10.4f}")

    print(f"\nРекомендуемая alpha (по overall NDCG@{args.k}): {best_alpha['alpha']:.2f}")
    if all(s["ndcg_keyword"] is not None for s in summary):
        best_kw = max(summary, key=lambda s: s["ndcg_keyword"])
        best_sem = max(summary, key=lambda s: s["ndcg_semantic"])
        print(f"  — лучшая для keyword-запросов: alpha={best_kw['alpha']:.2f}")
        print(f"  — лучшая для semantic-запросов: alpha={best_sem['alpha']:.2f}")
        if best_kw["alpha"] != best_sem["alpha"]:
            print("  Разные сегменты предпочитают разные alpha — рассмотрите query-adaptive alpha "
                  "(классификатор типа запроса перед вызовом /search) вместо одного глобального значения.")

    print(f"\nСохранено:\n  {detail_path}\n  {summary_path}\n  {chart_path}")

if __name__ == "__main__":
    main()
