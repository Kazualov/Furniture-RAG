

run:

python -m src.models.generate_embeddings \
  --input embeddings/office_products_micro.parquet \
  --image-dir data/images \
  --output-dir embeddings

python -m src.database.load_postgres
python -m src.database.load_qdrant
python -m src.search.main