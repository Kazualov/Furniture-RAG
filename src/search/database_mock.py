import asyncio
from typing import List
from src.search.interfaces import DBResultItem, ProductMetadata

# Mocking a keyword-heavy response from Postgres (BM25)
MOCK_POSTGRES_RESULTS = [
    DBResultItem(
        product_id="B00YQ6X8EO",
        score=12.45,
        metadata=ProductMetadata(
            asin="B00YQ6X8EO",
            parent_asin="B00YQ6X8EO",
            title="Scented Room Spray - Lavender & Chamomile",
            description="A soothing and relaxing room spray designed to freshen up your living space.",
            price=14.99,
            main_category="Home & Kitchen",
            average_rating=4.6,
            rating_number=1420
        )
    ),
    DBResultItem(
        product_id="B07XV12345",
        score=9.80,
        metadata=ProductMetadata(
            asin="B07XV12345",
            parent_asin="B07XV99999",
            title="Premium Lavender Scented Candle",
            description="Long-lasting soy wax candle infused with organic lavender essential oils.",
            price=18.50,
            main_category="Home & Kitchen",
            average_rating=4.2,
            rating_number=89
        )
    )
]

# Mocking a semantically relevant response from Qdrant (Dense)
MOCK_QDRANT_RESULTS = [
    DBResultItem(
        product_id="B088XYZ789",
        score=0.88,
        metadata=ProductMetadata(
            asin="B088XYZ789",
            parent_asin="B088XYZ789",
            title="Calming Aromatherapy Sleep Mist",
            description="Spray onto pillows and linens before bed. Contains lavender extracts.",
            price=12.00,
            main_category="Beauty & Personal Care",
            average_rating=4.7,
            rating_number=310
        )
    ),
    # Overlap item to test your Rank Fusion logic
    DBResultItem(
        product_id="B00YQ6X8EO",
        score=0.82,
        metadata=ProductMetadata(
            asin="B00YQ6X8EO",
            parent_asin="B00YQ6X8EO",
            title="Scented Room Spray - Lavender & Chamomile",
            description="A soothing and relaxing room spray designed to freshen up your living space.",
            price=14.99,
            main_category="Home & Kitchen",
            average_rating=4.6,
            rating_number=1420
        )
    )
]

class MockDBClient:
    @staticmethod
    async def search_sparse(query: str, limit: int = 10) -> List[DBResultItem]:
        await asyncio.sleep(0.04)
        return MOCK_POSTGRES_RESULTS[:limit]

    @staticmethod
    async def search_dense(query_vector: List[float], limit: int = 10) -> List[DBResultItem]:
        await asyncio.sleep(0.06)
        return MOCK_QDRANT_RESULTS[:limit]