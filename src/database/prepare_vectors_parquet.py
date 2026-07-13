import numpy as np
import pandas as pd

emb = np.load("embeddings/office_products_full_embeddings_fp32.npy")
meta = pd.read_parquet("embeddings/office_products_full.parquet")

df = pd.DataFrame({
    "parent_asin": meta["parent_asin"].values,
    "vector": [emb[i].tolist() for i in range(len(emb))],
})
df.to_parquet("data/vectors.parquet", index=False)
print(f"Wrote {len(df)} vectors → data/vectors.parquet")