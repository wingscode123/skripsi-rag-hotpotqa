from typing import List, Dict
from app.core.vector_store import VectorStore

class VectorRetriever:
    def __init__(self, index_name="hotpot_20k"):
        """
        Inisialisasi vector retriever.
        Memuat indeks FAISS yang sudah dibangun sebelumnya
        """
        self.store = VectorStore(index_name=index_name)
        try:
            self.store.load()
            print("[VectorRetriever] Index berhasil dimuat.")
        except Exception as e:
            print(f"[VectorRetriever] Error memuat index: {e}")
    
    def retrieve(self, query: str, top_k: int=5) -> List[Dict]:
        """
        Melakukan pencarian semantik (Semantic Search)
        Output: List of dict (text, score, metadata)
        """
        print(f"[VectorRetriever] Mencari: '{query}' (Top-{top_k})")
        results = self.store.search(query, top_k=top_k)
        return results