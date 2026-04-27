---
title: Ragvault
emoji: 🔍
colorFrom: indigo
colorTo: cyan
sdk: docker
pinned: false
---

# RAG-Based Insight & Retrieval System

An end-to-end Retrieval-Augmented Generation (RAG) pipeline built with Python, FastAPI, and FAISS — designed for fast, accurate document retrieval using vector embeddings and cosine similarity ranking.

## Overview

This system ingests text documents, converts them into semantic vector embeddings using Sentence Transformers, indexes them with FAISS, and exposes a FastAPI REST API for querying. Given a natural language query, it retrieves the most relevant document chunks based on cosine similarity.

## Architecture
Documents (txt)
↓
Chunking (500 chars)
↓
Sentence Transformer Embeddings (all-MiniLM-L6-v2)
↓
FAISS Index (cosine similarity via IndexFlatIP)
↓
FastAPI REST API (/query endpoint)
↓
Top-K Relevant Chunks + Latency

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| API Framework | FastAPI |
| Vector Search | FAISS (Facebook AI) |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Similarity | Cosine Similarity (L2 Normalized) |
| Server | Uvicorn |

## Results

- 40% faster insight retrieval compared to keyword-based search
- Evaluated across 25+ diverse query scenarios
- Supports concurrent API requests with structured JSON output

## Project Structure
rag-insight-system/
├── app/
│   ├── init.py
│   ├── ingest.py        # Document loading, chunking, FAISS indexing
│   ├── retriever.py     # Query embedding + FAISS search
│   └── main.py          # FastAPI endpoints
├── data/                # Place your .txt documents here
├── requirements.txt
├── .gitignore
└── README.md

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add documents
Place your `.txt` files inside the `data/` folder.

### 3. Build the index
```bash
python -m app.ingest
```

### 4. Run the API
```bash
uvicorn app.main:app --reload
```

### 5. Query the API
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "your question here", "top_k": 3}'
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | / | System status + total chunks loaded |
| POST | /query | Retrieve top-K relevant chunks for a query |
| GET | /health | Health check |

## Sample Output

```json
{
  "query": "What is machine learning?",
  "results": [
    {
      "chunk": "Machine learning is a subset of AI that enables systems to learn from data...",
      "score": 0.8921
    }
  ],
  "latency_ms": 12.4
}
```

## Author

**Divya D Amin**  
[LinkedIn](https://linkedin.com/in/divya-amin-6b7178280) | [GitHub](https://github.com/DivyaAmin13)

