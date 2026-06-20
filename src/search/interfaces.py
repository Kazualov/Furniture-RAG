from pydantic import BaseModel, Field
from typing import Optional

class ProductMetadata(BaseModel):
    asin: str = Field(..., description="Unique Amazon Standard Identification Number")
    parent_asin: str = Field(..., description="Parent ASIN for variant grouping")
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Detailed product description")
    price: Optional[float] = Field(None, description="Product price in USD")
    main_category: Optional[str] = Field(None, description="Primary product category")
    average_rating: Optional[float] = Field(None, description="Aggregated rating score (1-5)")
    rating_number: int = Field(0, description="Total number of ratings received")

class DBResultItem(BaseModel):
    product_id: str = Field(..., description="Maps to parent_asin or asin")
    score: float = Field(..., description="Raw match score from the underlying database")
    metadata: ProductMetadata

class SearchQueryRequest(BaseModel):
    query: str
    limit: int = 10
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Balance between Sparse (0.0) and Dense (1.0)")