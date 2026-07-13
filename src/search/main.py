import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
import io

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from PIL import Image
from sentence_transformers import SentenceTransformer

from src.database.database import LiveDatabaseClient
from src.models.interfaces import DBResultItem


@asynccontextmanager
async def lifespan(app: FastAPI):
    await LiveDatabaseClient.connect()
    yield
    await LiveDatabaseClient.disconnect()


app = FastAPI(title="Multimodal Hybrid Search Engine API", version="0.2.0", lifespan=lifespan)

print("Loading Multimodal CLIP Encoder...")
# Загружаем CLIP глобально
encoder_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")


def weighted_reciprocal_rank_fusion(
        sparse_results: List[DBResultItem],
        dense_results: List[DBResultItem],
        alpha: float = 0.5,
        k: int = 60
) -> List[Dict[str, Any]]:
    # Ваша текущая рабочая функция RRF без изменений
    rrf_scores: Dict[str, float] = {}
    metadata_map: Dict[str, Any] = {}

    sparse_weight = alpha
    dense_weight = 1.0 - alpha

    for rank, item in enumerate(sparse_results, start=1):
        pid = item.product_id
        if pid not in rrf_scores:
            rrf_scores[pid] = 0.0
            metadata_map[pid] = item.metadata.model_dump()
        rrf_scores[pid] += sparse_weight * (1.0 / (k + rank))

    for rank, item in enumerate(dense_results, start=1):
        pid = item.product_id
        if pid not in rrf_scores:
            rrf_scores[pid] = 0.0
            metadata_map[pid] = item.metadata.model_dump()
        rrf_scores[pid] += dense_weight * (1.0 / (k + rank))

    sorted_pids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {"product_id": pid, "rrf_score": round(score, 6), "metadata": metadata_map[pid]}
        for pid, score in sorted_pids
    ]


@app.post("/search")
async def multimodal_search(
        query: Optional[str] = Form(None),
        file: Optional[UploadFile] = File(None),
        limit: int = Form(10),
        alpha: float = Form(0.5)
):
    if not query and not file:
        raise HTTPException(status_code=400, detail="Предоставьте текст запроса (query) или изображение (file).")

    try:
        text_vector = None
        image_vector = None

        # 1. Обработка текстового запроса
        if query:
            text_vector = encoder_model.encode(query, normalize_embeddings=True)

        # 2. Обработка входящего изображения
        if file:
            image_bytes = await file.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_vector = encoder_model.encode(pil_image, normalize_embeddings=True)

        # 3. Нахождение результирующего вектора для Qdrant
        if text_vector is not None and image_vector is not None:
            # Текст + Фото: Линейная комбинация векторов (50/50 или можно настроить баланс)
            combined_vector = (text_vector * 0.5) + (image_vector * 0.5)
            # Повторно нормализуем вектор
            query_vector = (combined_vector / np.linalg.norm(combined_vector)).tolist()
        elif text_vector is not None:
            query_vector = text_vector.tolist()
        else:
            query_vector = image_vector.tolist()

        # 4. Выполнение поиска в БД
        # Поиск в Qdrant работает всегда, независимо от типа запроса
        dense_task = LiveDatabaseClient.search_dense(query_vector, limit=limit)

        # В Postgres идем только если есть текст. Если только фото — sparse поиск пропускаем
        if query:
            sparse_task = LiveDatabaseClient.search_sparse(query, limit=limit)
            sparse_results, dense_results = await asyncio.gather(sparse_task, dense_task)

            # Сливаем результаты через RRF
            final_results = weighted_reciprocal_rank_fusion(
                sparse_results=sparse_results,
                dense_results=dense_results,
                alpha=alpha
            )
        else:
            # Чисто визуальный поиск: берем топ из Qdrant и приводим к общему виду ответа
            dense_results = await dense_task
            final_results = [
                {
                    "product_id": item.product_id,
                    "rrf_score": round(item.score, 6),
                    "metadata": item.metadata.model_dump()
                }
                for item in dense_results
            ]

        return final_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Multimodal Engine Error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("src.search.main:app", host="0.0.0.0", port=8000, reload=True)