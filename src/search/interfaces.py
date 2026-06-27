import math
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class ProductMetadata(BaseModel):
    parent_asin: str = Field(..., description="Parent ASIN for variant grouping")
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Detailed product description")
    features: Optional[str] = Field(None, description="Product features/bullet points")
    categories: Optional[str] = Field(None, description="Delimited subcategory string")
    details_text: Optional[str] = Field(None, description="Converted details dictionary to text")
    price: Optional[float] = Field(None, description="Product price in USD")
    average_rating: Optional[float] = Field(None, description="Aggregated rating score (1-5)")
    rating_number: Optional[int] = Field(0, description="Total number of ratings received")
    store: Optional[str] = Field(None, description="Store brand information")
    image_url: Optional[str] = Field(None, description="Extracted display image URL")
    full_text: Optional[str] = Field(None, description="Concatenated rich text used for embeddings")

    @field_validator('price', 'average_rating', 'rating_number', mode='before')
    @classmethod
    def clean_missing_numeric_data(cls, v):
        """
        Intercepts raw data inputs before validation to safely convert
        stringified 'None', 'NaN', or empty fields into true Python None.
        """
        if isinstance(v, str):
            cleaned = v.strip().lower()
            if cleaned in ('none', 'null', 'nan', ''):
                return None

        # Handle native floating-point NaNs that leak from Pandas dataframes
        if isinstance(v, float) and math.isnan(v):
            return None

        return v


class DBResultItem(BaseModel):
    product_id: str = Field(..., description="Maps to parent_asin")
    score: float = Field(..., description="Raw match score from the underlying database")
    metadata: ProductMetadata


class SearchQueryRequest(BaseModel):
    query: str
    limit: int = 10
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Balance between Sparse (0.0) and Dense (1.0)")