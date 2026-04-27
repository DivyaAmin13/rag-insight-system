from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import time
import os

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="RAG Insight & Retrieval System",
    description="End-to-end RAG pipeline with FAISS vector search and FastAPI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load index only if it exists
index, documents = None, []
if os.path.exists("data/index.faiss") and os.path.exists("data/documents.pkl"):
    from app.retriever import load_index
    index, documents = load_index()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    query: str
    results: list
    latency_ms: float

@app.get("/")
def root():
    return FileResponse("app/static/index.html")

@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    if index is None:
        raise HTTPException(status_code=503, detail="Index not loaded. Please run ingest first.")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    from app.retriever import retrieve
    start = time.time()
    results = retrieve(request.query, index, documents, request.top_k)
    latency = round((time.time() - start) * 1000, 2)
    return QueryResponse(query=request.query, results=results, latency_ms=latency)

@app.get("/health")
def health():
    return {"status": "healthy", "chunks_loaded": len(documents), "index_loaded": index is not None}