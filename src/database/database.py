import os
import asyncpg
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import SearchParams

# 1. Update imports for the new folder structure
from src.models.interfaces import DBResultItem, ProductMetadata

# 2. Import teammate's config to stay synced with indexing
from src.indexing import config


class LiveDatabaseClient:
    _pg_pool: asyncpg.Pool | None = None
    _qdrant: AsyncQdrantClient | None = None

    @classmethod
    async def connect(cls):
        """Initialize connection pools. Call this on app startup."""
        if cls._pg_pool is None:
            # Connect using the centralized Postgres DSN
            cls._pg_pool = await asyncpg.create_pool(config.POSTGRES_DSN)

        if cls._qdrant is None:
            # Connect using config variables (with fallbacks just in case)
            host = getattr(config, "QDRANT_HOST", os.getenv("QDRANT_HOST", "localhost"))
            port = getattr(config, "QDRANT_HTTP_PORT", int(os.getenv("QDRANT_HTTP_PORT", 6333)))
            cls._qdrant = AsyncQdrantClient(host=host, port=port)

        print("Live DB connections established.")

    @classmethod
    async def disconnect(cls):
        """Cleanly close connections."""
        if cls._pg_pool:
            await cls._pg_pool.close()
        if cls._qdrant:
            await cls._qdrant.close()

    @classmethod
    async def search_sparse(cls, query: str, limit: int = 10) -> list[DBResultItem]:
        """PostgreSQL Full-Text Search using the precomputed search_vector."""
        if not cls._pg_pool:
            await cls.connect()

        # 3. Swap on-the-fly to_tsvector with the teammate's indexed search_vector
        sql = """
              SELECT *, \
                     ts_rank_cd(search_vector, ts_query) AS score
              FROM products, \
                   websearch_to_tsquery($1, $2) AS ts_query
              WHERE search_vector @@ ts_query
              ORDER BY score DESC
                  LIMIT $3; \
              """

        async with cls._pg_pool.acquire() as conn:
            # Pass the configured language from src.indexing.config
            rows = await conn.fetch(sql, config.FTS_LANGUAGE, query, limit)

        results = []
        for row in rows:
            row_dict = dict(row)
            score = row_dict.pop("score", 0.0)

            # Pop the migration columns so they don't break Pydantic validation
            row_dict.pop("search_vector", None)
            row_dict.pop("ts_query", None)

            results.append(DBResultItem(
                product_id=str(row_dict["parent_asin"]),
                score=float(score),
                metadata=ProductMetadata(**row_dict)
            ))

        return results

    @classmethod
    async def search_dense(
            cls,
            query_vector: list[float],
            limit: int = 10,
            exact: bool = False  # <-- Новый флаг для байпаса индекса
    ) -> list[DBResultItem]:
        """Qdrant Vector Search + Postgres Metadata Hydration."""
        if not cls._qdrant or not cls._pg_pool:
            await cls.connect()

        # 4. Target the centralized Qdrant collection name
        qdrant_response = await cls._qdrant.query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True,
            search_params=SearchParams(exact=exact)  # <-- Передаем флаг в Qdrant
        )
        qdrant_results = qdrant_response.points

        if not qdrant_results:
            return []

        product_scores = {
            str(hit.payload.get("product_id", hit.id)): hit.score
            for hit in qdrant_results
        }
        product_ids = list(product_scores.keys())

        sql = "SELECT * FROM products WHERE parent_asin = ANY($1::text[])"
        async with cls._pg_pool.acquire() as conn:
            rows = await conn.fetch(sql, product_ids)

        results = []
        for row in rows:
            row_dict = dict(row)
            pid = str(row_dict["parent_asin"])

            # Pop migration columns here as well just to be safe
            row_dict.pop("search_vector", None)

            results.append(DBResultItem(
                product_id=pid,
                score=float(product_scores[pid]),
                metadata=ProductMetadata(**row_dict)
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results
