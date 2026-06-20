import asyncio
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from src.search.interfaces import SearchQueryRequest, DBResultItem
from src.search.database_mock import (MockDBClient)

app = FastAPI(title="Hybrid Search Engine API", version="0.1.0")

EMBEDDING_DIMENSIONALITY = 384

# Temporary mock embedding function while Azamat optimizes the embedding pipeline
def get_mock_embedding(text: str) -> List[float]:
    # Return a dummy vector with fixed dimensions (e.g., 384 for all-MiniLM-L6-v2)
    return [0.1] * EMBEDDING_DIMENSIONALITY


@app.post("/search", response_model=List[Dict[str, Any]])
async def hybrid_search(payload: SearchQueryRequest):
    try:
        query_text = payload.query

        # 1. Generate the query embedding (currently mocked)
        query_vector = get_mock_embedding(query_text)

        # 2. Query both databases concurrently using Egor's interface wrappers
        # Using gather avoids sequential latency bottlenecks
        sparse_task = MockDBClient.search_sparse(query_text, limit=payload.limit)
        dense_task = MockDBClient.search_dense(query_vector, limit=payload.limit)

        sparse_results, dense_results = await asyncio.gather(sparse_task, dense_task)

        # 3. Rank Fusion Logic (Your core task for Days 4-7)
        # Currently using a basic merge fallback to ensure the endpoint functions
        final_results = reciprocal_rank_fusion(sparse_results, dense_results)

        return final_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Search Error: {str(e)}")


def reciprocal_rank_fusion(
        sparse_results: List[DBResultItem],
        dense_results: List[DBResultItem],
        k: int = 60
) -> List[Dict[str, Any]]:
    """
    Applies Reciprocal Rank Fusion (RRF) to merge keyword and semantic search results.
    Score = sum(1 / (k + rank)) for each appearance in a result list.
    """
    rrf_scores: Dict[str, float] = {}
    metadata_map: Dict[str, Any] = {}

    # Process Sparse (Postgres) ranks
    # Assumes Egor's output is already sorted by score descending
    for rank, item in enumerate(sparse_results, start=1):
        pid = item.product_id
        if pid not in rrf_scores:
            rrf_scores[pid] = 0.0
            metadata_map[pid] = item.metadata.model_dump()
        rrf_scores[pid] += 1.0 / (k + rank)

    # Process Dense (Qdrant) ranks
    for rank, item in enumerate(dense_results, start=1):
        pid = item.product_id
        if pid not in rrf_scores:
            rrf_scores[pid] = 0.0
            metadata_map[pid] = item.metadata.model_dump()
        rrf_scores[pid] += 1.0 / (k + rank)

    # Sort items based on their aggregated RRF score in descending order
    sorted_pids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Format the final output for your API response
    final_ranked_results = [
        {
            "product_id": pid,
            "rrf_score": round(score, 6),
            "metadata": metadata_map[pid]
        }
        for pid, score in sorted_pids
    ]

    return final_ranked_results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)