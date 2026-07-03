import numpy as np
import pandas as pd

emb = np.load("src/models/embeddings/office_products_micro_embeddings_fp32.npy")
meta = pd.read_parquet("src/models/embeddings/office_products_micro.parquet")

df = pd.DataFrame({
    "parent_asin": meta["parent_asin"].values,
    "vector": [emb[i].tolist() for i in range(len(emb))],
})
df.to_parquet("data/vectors.parquet", index=False)
print(f"Wrote {len(df)} vectors → data/vectors.parquet")