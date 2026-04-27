from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.retriever import load_index, retrieve
import time

app = FastAPI(
    title="RAG Insight & Retrieval System",
    description="End-to-end RAG pipeline with FAISS vector search and FastAPI",
    version="1.0.0"
)

# Load index on startup
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
    return {"message": "RAG Insight System is running", "total_chunks": len(documents)}

@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    start = time.time()
    results = retrieve(request.query, index, documents, request.top_k)
    latency = round((time.time() - start) * 1000, 2)
    
    return QueryResponse(
        query=request.query,
        results=results,
        latency_ms=latency
    )

@app.get("/health")
def health():
    return {"status": "healthy", "chunks_loaded": len(documents)}