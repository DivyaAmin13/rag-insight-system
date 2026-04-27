import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_documents(data_dir: str = "data"):
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                text = f.read()
                chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                documents.extend(chunks)
    return documents

def build_index(documents):
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def save_index(index, documents, index_path="data/index.faiss", docs_path="data/documents.pkl"):
    faiss.write_index(index, index_path)
    with open(docs_path, "wb") as f:
        pickle.dump(documents, f)
    print(f"Index saved: {index.ntotal} chunks indexed.")

if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()
    print(f"Total chunks: {len(docs)}")
    index, _ = build_index(docs)
    save_index(index, docs)