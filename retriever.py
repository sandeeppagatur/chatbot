# retriever.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class ContextRetriever:
    def __init__(self, docs, model_name="all-MiniLM-L6-v2"):
        self.docs = docs
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(docs, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def retrieve(self, query, top_k=2):
        query_vec = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, top_k)
        results = [self.docs[i] for i in indices[0]]
        print(f"Retrieved context: {results}")
        return results
