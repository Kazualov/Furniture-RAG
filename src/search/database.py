import os
import asyncpg
from qdrant_client import AsyncQdrantClient
from dotenv import load_dotenv
from src.search.interfaces import DBResultItem, ProductMetadata

load_dotenv()

POSTGRES_DSN = (
    f"postgresql://{os.getenv('POSTGRES_USER', 'rag_user')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'rag_password')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', 5432)}/"
    f"{os.getenv('POSTGRES_DB', 'furniture_db')}"
)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_HTTP_PORT = int(os.getenv("QDRANT_HTTP_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "furniture_products")


class LiveDatabaseClient:
    _pg_pool: asyncpg.Pool | None = None
    _qdrant: AsyncQdrantClient | None = None

    @classmethod
    async def connect(cls):
        """Initialize connection pools. Call this on app startup."""
        if cls._pg_pool is None:
            cls._pg_pool = await asyncpg.create_pool(POSTGRES_DSN)
        if cls._qdrant is None:
            cls._qdrant = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_HTTP_PORT)
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
        """PostgreSQL Full-Text Search."""
        if not cls._pg_pool:
            await cls.connect()

        # Basic Postgres full-text search. Concat title and store for a wider net.
        sql = """
              SELECT *, \
                     ts_rank_cd( \
                             to_tsvector('english', concat_ws(' ', title, store)), \
                             websearch_to_tsquery('english', $1) \
                     ) AS score
              FROM products
              WHERE to_tsvector('english', concat_ws(' ', title, store)) @@ websearch_to_tsquery('english' \
                  , $1)
              ORDER BY score DESC
                  LIMIT $2; \
              """

        async with cls._pg_pool.acquire() as conn:
            rows = await conn.fetch(sql, query, limit)

        results = []
        for row in rows:
            row_dict = dict(row)
            score = row_dict.pop("score", 0.0)

            # Double-check that your TRELLIS-500K metadata keys align here.
            # If you standardized on file_identifier instead of parent_asin later,
            # you'll need to map it accordingly.
            results.append(DBResultItem(
                product_id=str(row_dict["parent_asin"]),
                score=float(score),
                metadata=ProductMetadata(**row_dict)
            ))

        return results

    @classmethod
    async def search_dense(cls, query_vector: list[float], limit: int = 10) -> list[DBResultItem]:
        """Qdrant Vector Search + Postgres Metadata Hydration."""
        if not cls._qdrant or not cls._pg_pool:
            await cls.connect()

        # 1. Fetch nearest neighbors from Qdrant
        qdrant_response = await cls._qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            with_payload=True
        )
        qdrant_results = qdrant_response.points

        if not qdrant_results:
            return []

        # 2. Extract IDs and map their cosine similarity scores
        product_scores = {
            str(hit.payload.get("product_id", hit.id)): hit.score
            for hit in qdrant_results
        }
        product_ids = list(product_scores.keys())

        # 3. Hydrate metadata from Postgres using the retrieved IDs
        sql = "SELECT * FROM products WHERE parent_asin = ANY($1::text[])"
        async with cls._pg_pool.acquire() as conn:
            rows = await conn.fetch(sql, product_ids)

        results = []
        for row in rows:
            row_dict = dict(row)
            pid = str(row_dict["parent_asin"])
            results.append(DBResultItem(
                product_id=pid,
                score=float(product_scores[pid]),
                metadata=ProductMetadata(**row_dict)
            ))

        # 4. Postgres returns rows in an arbitrary order with `ANY()`.
        # We must re-sort them to match Qdrant's vector distances.
        results.sort(key=lambda x: x.score, reverse=True)
        return results