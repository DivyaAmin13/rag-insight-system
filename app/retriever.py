import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_index(index_path="data/index.faiss", docs_path="data/documents.pkl"):
    index = faiss.read_index(index_path)
    with open(docs_path, "rb") as f:
        documents = pickle.load(f)
    return index, documents

def retrieve(query: str, index, documents, top_k: int = 3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)
    
    scores, indices = index.search(query_embedding, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "chunk": documents[idx],
            "score": round(float(score), 4)
        })
    return results