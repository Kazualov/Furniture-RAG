import asyncio
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from src.search.interfaces import SearchQueryRequest, DBResultItem
from src.search.database_mock import MockDBClient

app = FastAPI(title="Hybrid Search Engine API", version="0.1.0")

EMBEDDING_DIMENSIONALITY = 384

def get_mock_embedding(text: str) -> List[float]:
    return [0.1] * EMBEDDING_DIMENSIONALITY


def weighted_reciprocal_rank_fusion(
        sparse_results: List[DBResultItem],
        dense_results: List[DBResultItem],
        alpha: float = 0.5,
        k: int = 60
) -> List[Dict[str, Any]]:
    """
    Applies Weighted Reciprocal Rank Fusion.

    alpha scales the keyword search impact.
    (1 - alpha) scales the semantic vector search impact.
    """
    rrf_scores: Dict[str, float] = {}
    metadata_map: Dict[str, Any] = {}

    # Weight for Sparse results
    sparse_weight = alpha
    # Weight for Dense results
    dense_weight = 1.0 - alpha

    # Process Sparse (Postgres) ranks
    for rank, item in enumerate(sparse_results, start=1):
        pid = item.product_id
        if pid not in rrf_scores:
            rrf_scores[pid] = 0.0
            metadata_map[pid] = item.metadata.model_dump()
        rrf_scores[pid] += sparse_weight * (1.0 / (k + rank))

    # Process Dense (Qdrant) ranks
    for rank, item in enumerate(dense_results, start=1):
        pid = item.product_id
        if pid not in rrf_scores:
            rrf_scores[pid] = 0.0
            metadata_map[pid] = item.metadata.model_dump()
        rrf_scores[pid] += dense_weight * (1.0 / (k + rank))

    # Re-rank everything based on the weighted scores
    sorted_pids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {
            "product_id": pid,
            "rrf_score": round(score, 6),
            "metadata": metadata_map[pid]
        }
        for pid, score in sorted_pids
    ]


@app.post("/search", response_model=List[Dict[str, Any]])
async def hybrid_search(payload: SearchQueryRequest):
    try:
        query_text = payload.query

        query_vector = get_mock_embedding(query_text)

        sparse_task = MockDBClient.search_sparse(query_text, limit=payload.limit)
        dense_task = MockDBClient.search_dense(query_vector, limit=payload.limit)
        sparse_results, dense_results = await asyncio.gather(sparse_task, dense_task)

        # Pass the payload's alpha down to the fusion logic
        final_results = weighted_reciprocal_rank_fusion(
            sparse_results=sparse_results,
            dense_results=dense_results,
            alpha=payload.alpha
        )
        return final_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Search Error: {str(e)}")