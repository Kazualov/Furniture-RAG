"""
Smoke-тест: проверяет, что лексический (Postgres) и векторный (Qdrant) индексы
работают end-to-end. Самодостаточный — создаёт временные данные и убирает их.

    python -m src.indexing.smoke_test

Код выхода 0 — всё прошло; 1 — есть провалившиеся проверки.
"""

import asyncio
import sys

import numpy as np

from src.indexing import config


# --------------------------------------------------------------------------- #
# Постгрес: лексический индекс
# --------------------------------------------------------------------------- #
SMOKE_ROWS = [
    # (asin, parent_asin, title, description)
    ("SMOKE_1", "SMOKE_1", "Black Leather Office Chair",
     "Ergonomic executive chair with lumbar support."),
    ("SMOKE_2", "SMOKE_2", "Wooden Dining Table",
     "Solid oak table for six people."),
    ("SMOKE_3", "SMOKE_3", "Office Desk Lamp",
     "LED desk lamp with adjustable brightness."),
]


async def test_postgres() -> bool:
    import asyncpg
    from src.indexing.apply_postgres_migrations import load_migrations

    print("\n[Postgres] подключение ...")
    conn = await asyncpg.connect(config.POSTGRES_DSN)
    try:
        # 1) применяем миграцию (идемпотентно)
        for name, sql in load_migrations():
            await conn.execute(sql)
        print("[Postgres] миграция применена (search_vector + GIN)")

        # 2) вставляем тестовые строки
        await conn.executemany(
            """
            INSERT INTO products (asin, parent_asin, title, description, rating_number)
            VALUES ($1, $2, $3, $4, 0)
            ON CONFLICT (asin) DO UPDATE SET
                title = EXCLUDED.title, description = EXCLUDED.description
            """,
            SMOKE_ROWS,
        )

        # 3) лексический поиск — ожидаем, что "office chair" поднимет SMOKE_1 наверх
        rows = await conn.fetch(
            """
            SELECT asin, ts_rank_cd(search_vector, query) AS score
            FROM products, websearch_to_tsquery($1, $2) AS query
            WHERE search_vector @@ query AND asin LIKE 'SMOKE_%'
            ORDER BY score DESC
            """,
            config.FTS_LANGUAGE, "office chair",
        )
        found = [r["asin"] for r in rows]
        print(f"[Postgres] поиск 'office chair' -> {found}")

        ok = bool(found) and found[0] == "SMOKE_1"
        if ok:
            print("[Postgres] PASS: индекс ранжирует, топ-1 = SMOKE_1")
        else:
            print("[Postgres] FAIL: ожидался SMOKE_1 на первом месте")
        return ok
    finally:
        # cleanup
        await conn.execute("DELETE FROM products WHERE asin LIKE 'SMOKE_%'")
        await conn.close()


# --------------------------------------------------------------------------- #
# Qdrant: векторный индекс HNSW
# --------------------------------------------------------------------------- #
def test_qdrant() -> bool:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    from src.indexing.qdrant_collection import DEFAULT_HNSW, ensure_collection

    name = "smoke_hnsw"
    print("\n[Qdrant] подключение ...")
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_HTTP_PORT)

    try:
        # на чистый старт
        if name in [c.name for c in client.get_collections().collections]:
            client.delete_collection(name)

        # 1) создаём коллекцию с нашим HNSW-конфигом
        ensure_collection(client, hnsw=DEFAULT_HNSW, collection_name=name)
        print(f"[Qdrant] коллекция '{name}' создана с HNSW "
              f"m={DEFAULT_HNSW.m}, ef_construct={DEFAULT_HNSW.ef_construct}")

        # 2) заливаем тестовые нормированные вектора
        rng = np.random.default_rng(0)
        n = 64
        vecs = rng.standard_normal((n, config.EMBEDDING_DIM)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        client.upsert(
            collection_name=name,
            points=[PointStruct(id=i, vector=vecs[i].tolist()) for i in range(n)],
            wait=True,
        )
        print(f"[Qdrant] залито {n} тестовых векторов (dim={config.EMBEDDING_DIM})")

        # 3) поиск: запрос вектором точки #7 -> сама точка должна быть топ-1
        resp = client.query_points(collection_name=name, query=vecs[7].tolist(), limit=5)
        ids = [p.id for p in resp.points]
        print(f"[Qdrant] поиск по вектору точки #7 -> {ids}")

        ok = len(ids) == 5 and ids[0] == 7
        if ok:
            print("[Qdrant] PASS: поиск возвращает k результатов, топ-1 = сам запрос")
        else:
            print("[Qdrant] FAIL: ожидалась точка #7 на первом месте и 5 результатов")
        return ok
    finally:
        if name in [c.name for c in client.get_collections().collections]:
            client.delete_collection(name)


# --------------------------------------------------------------------------- #
def main() -> None:
    results = {}
    try:
        results["postgres"] = asyncio.run(test_postgres())
    except Exception as e:  # noqa: BLE001
        print(f"[Postgres] ERROR: {type(e).__name__}: {e}")
        results["postgres"] = False

    try:
        results["qdrant"] = test_qdrant()
    except Exception as e:  # noqa: BLE001
        print(f"[Qdrant] ERROR: {type(e).__name__}: {e}")
        results["qdrant"] = False

    print("\n==================== ИТОГ ====================")
    for k, v in results.items():
        print(f"  {k:<10} {'PASS' if v else 'FAIL'}")
    print("=============================================")

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
