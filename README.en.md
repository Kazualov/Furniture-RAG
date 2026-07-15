# RAG System for Office Products Search

A hybrid RAG system for semantic and full-text search over an office furniture and supplies catalog, built on a subset of the **Amazon Open Source Dataset** (*Office Furniture* category, ~700,000 product listings).

The project demonstrates a step-by-step evolution of a search system: from a naive exact-vector-search baseline to a full hybrid architecture with multi-signal ranking.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Version Evolution](#version-evolution)
- [Ranking Method: Weighted RRF](#ranking-method-weighted-rrf)

## Overview

The goal of this project is to build a fast and accurate system for retrieving relevant products from a natural-language query. It combines:

- **Dense embeddings** for semantic similarity between the query and documents;
- **Sparse BM25 retrieval** for precise lexical matching (SKUs, brands, model names);
- **Weighted Reciprocal Rank Fusion (RRF)** to merge both ranked result lists into a single, more robust ranking than either method alone.

## Dataset

- **Source:** Amazon Open Source Dataset
- **Category:** Office Furniture
- **Size:** ~700,000 product records
- Product text fields (title, description, attributes) are used to build both the vector index and the full-text index.

## Architecture

The system relies on two data stores working together:

| Component | Purpose |
|---|---|
| **Qdrant** | Vector database storing and searching dense embeddings (384 dimensions), using HNSW indexing and INT8 quantization |
| **PostgreSQL** | Relational storage for product data and BM25 full-text search |

Query flow:

1. A user query is processed in parallel by two pipelines: dense search in Qdrant and BM25 search in PostgreSQL.
2. Both ranked result lists are passed into the **Weighted RRF** module.
3. The final result list is produced by weighted rank fusion using an optimal fusion coefficient **α**.

## Tech Stack

- **Vector database:** Qdrant (HNSW index, INT8 quantization from FP32)
- **Relational database:** PostgreSQL (BM25 full-text search)
- **Embeddings:** dense vectors, 384 dimensions
- **Ranking:** Weighted Reciprocal Rank Fusion (RRF)

## Version Evolution

The project was built iteratively, with each version addressing the main bottleneck of the previous one (index accuracy → speed → memory → ranking quality):

### v1.0 — Baseline / Naive Approach
Pure vector search using an **exact (Flat) index** and **heavy FP32 vectors**.
A full brute-force scan with no approximation — the accuracy reference point, but the slowest and most memory-intensive variant.

### v1.1 — Database Optimization
Switched to an **approximate HNSW index** while keeping **FP32** vectors.
Significant search speed gains from graph-based indexing, at a minor accuracy cost relative to the baseline.

### v1.2 — Model Optimization
**HNSW index + quantized vectors (INT8)**.
Reduced memory footprint and further speed gains by converting embeddings from FP32 to an INT8 representation.

### v2.0 — Final Hybrid System
**BM25 (PostgreSQL) + HNSW INT8 (Qdrant) + Rank Fusion** with a tuned optimal α.
Combining lexical and semantic search yields the most relevant and robust results, while preserving the performance gains achieved in the previous versions.

## Ranking Method: Weighted RRF

The final ranking is produced using **Weighted Reciprocal Rank Fusion** — a method for merging multiple ranked result lists (BM25 and dense search) into a single ranking. The weight **α** controls the relative contribution of each source and is tuned empirically to achieve the best retrieval quality.

---

*This README describes the project's architecture and development stages. Setup and usage instructions can be added in a dedicated section as the project's infrastructure matures.*
