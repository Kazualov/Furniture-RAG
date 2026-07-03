"""
Лексический поиск поверх tsvector + GIN. Ранжирование через ts_rank_cd.

    python -m src.indexing.search_postgres "black leather office chair"
"""

import asyncio
import sys

import asyncpg

from src.indexing import config

# websearch_to_tsquery безопасно парсит свободный пользовательский ввод.
SEARCH_QUERY = """
    SELECT
        asin            AS product_id,
        title,
        price,
        ts_rank_cd(search_vector, query) AS score
    FROM products, websearch_to_tsquery($1, $2) AS query
    WHERE search_vector @@ query
    ORDER BY score DESC
    LIMIT $3;
"""


async def search_sparse(query: str, limit: int = 10) -> list[dict]:
    conn = await asyncpg.connect(config.POSTGRES_DSN)
    try:
        rows = await conn.fetch(SEARCH_QUERY, config.FTS_LANGUAGE, query, limit)
        return [dict(r) for r in rows]
    finally:
        await conn.close()


async def _main() -> None:
    query = " ".join(sys.argv[1:]) or "office chair"
    print(f"Запрос: {query!r} (словарь '{config.FTS_LANGUAGE}')\n")
    results = await search_sparse(query)
    if not results:
        print("Ничего не найдено. Проверь, что данные залиты и миграция применена.")
        return
    for i, r in enumerate(results, 1):
        print(f"{i:>2}. score={r['score']:.4f}  [{r['product_id']}]  {r['title']}")


if __name__ == "__main__":
    asyncio.run(_main())
