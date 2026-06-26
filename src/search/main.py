import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from src.search.interfaces import SearchQueryRequest, DBResultItem
from src.search.database_mock import RealDataLocalSimulator

app = FastAPI(title="Hybrid Search Engine API", version="0.1.0")

# Initialize the real encoder model globally so it only loads once into memory on startup
print("Loading text encoder model (all-MiniLM-L6-v2)...")
encoder_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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

        # 1. Generate real query embeddings using SentenceTransformers
        # Convert to numpy array and ensure it is flat
        query_vector = encoder_model.encode(query_text, normalize_embeddings=True).tolist()

        # 2. Query our high-fidelity local simulator
        sparse_task = RealDataLocalSimulator.search_sparse(query_text, limit=payload.limit)
        dense_task = RealDataLocalSimulator.search_dense(query_vector, limit=payload.limit)
        sparse_results, dense_results = await asyncio.gather(sparse_task, dense_task)

        # 3. Fuse real product listings using your RRF logic
        final_results = weighted_reciprocal_rank_fusion(
            sparse_results=sparse_results,
            dense_results=dense_results,
            alpha=payload.alpha
        )
        return final_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Engine Error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)