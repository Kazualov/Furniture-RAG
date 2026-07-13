import json
import math
import requests
from metrics import ndcg, recall_at_k, reciprocal_rank

API_URL = "http://localhost:8000/search"
TOP_K = 10

def search(query, alpha=0.5, limit=10):

    response = requests.post(
        API_URL,
        json={
            "query": query,
            "alpha": alpha,
            "limit": limit
        }
    )

    response.raise_for_status()

    data = response.json()

    return [item["product_id"] for item in data]


def evaluate(golden_dataset):

    recall_scores = []
    mrr_scores = []
    ndcg_scores = []

    for sample in golden_dataset:

        prediction = search(sample["query"])

        relevant = sample["relevant_product_ids"]

        recall_scores.append(
            recall_at_k(relevant, prediction, TOP_K)
        )

        mrr_scores.append(
            reciprocal_rank(relevant, prediction)
        )

        ndcg_scores.append(
            ndcg(relevant, prediction, TOP_K)
        )

    print(f"Recall@{TOP_K}: {sum(recall_scores)/len(recall_scores):.4f}")
    print(f"MRR:           {sum(mrr_scores)/len(mrr_scores):.4f}")
    print(f"NDCG@{TOP_K}:  {sum(ndcg_scores)/len(ndcg_scores):.4f}")


if __name__ == "__main__":

    with open("golden_dataset.json", encoding="utf-8") as f:
        golden_dataset = json.load(f)

    evaluate(golden_dataset)