from app.core.retriever_vector import VectorRetriever
from app.core.retriever_graph import GraphRetriever

class HybridRetriever:
    def __init__(self):
        self.vector_retriever = VectorRetriever(index_name="hotpot_20k")
        self.graph_retriever = GraphRetriever(graph_file="knowledge_graph_20k.pkl")

    def retrieve(self, query: str, top_k: int = 5, alpha: float = 0.5):
        """
        Hybrid Retrieval dengan mekanisme 'Smart Fallback'.
        Strategi: 
        1. Memprioritaskan reasoning eksplisit dari Graph (High Precision).
        2. Menggunakan Vector untuk melengkapi konteks atau sebagai cadangan jika Graph kosong (High Recall).
        """
        combined_results = []
        seen_texts = set()

        # Mengambil dari Graph (Coba ambil 50% dari jatah top_k)
        graph_results = self.graph_retriever.retrieve(query)
        
        # Filter hasil graph yang duplikat atau kosong
        valid_graph_results = []
        for item in graph_results:
            # Normalisasi sederhana untuk deduplikasi
            sig = item['text'].lower().strip()[:50]
            if sig not in seen_texts:
                valid_graph_results.append(item)
                seen_texts.add(sig)

        # Menentukan berapa slot tersisa untuk Vector
        num_graph_used = len(valid_graph_results)
        target_vector_count = max(2, top_k - num_graph_used)
        
        if num_graph_used == 0:
            target_vector_count = top_k

        # Mengambil dari Vector
        vec_results = self.vector_retriever.retrieve(query, top_k=target_vector_count + 2) # Ambil lebih untuk cadangan
        
        valid_vector_results = []
        for item in vec_results:
            sig = item['text'].lower().strip()[:50]
            if sig not in seen_texts:
                valid_vector_results.append(item)
                seen_texts.add(sig)

        # Menggabungkan: Graph (Reasoning) + Vector (Context)
        combined_results = valid_graph_results[:top_k] + valid_vector_results
        return combined_results[:top_k]