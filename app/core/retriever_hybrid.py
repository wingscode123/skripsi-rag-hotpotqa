from app.core.retriever_vector import VectorRetriever
from app.core.retriever_graph import GraphRetriever

class HybridRetriever:
    def __init__(self):
        self.vector_retriever = VectorRetriever(index_name="hotpot_v1")
        self.graph_retriever = GraphRetriever(graph_file="knowledge_graph.pkl")

    def retrieve(self, query: str, top_k: int = 5, alpha: float = 0.5):
        """
        Hybrid Retrieval dengan mekanisme 'Smart Fallback'.
        Strategi: 
        1. Prioritaskan reasoning eksplisit dari Graph (High Precision).
        2. Gunakan Vector untuk melengkapi konteks atau sebagai cadangan jika Graph kosong (High Recall).
        """
        combined_results = []
        seen_texts = set()

        # 1. Ambil dari Graph (Coba ambil 50% dari jatah top_k)
        # Kita ingin jawaban yang 'pasti' dulu
        graph_results = self.graph_retriever.retrieve(query)
        
        # Filter hasil graph yang duplikat atau kosong
        valid_graph_results = []
        for item in graph_results:
            # Normalisasi sederhana untuk deduplikasi
            sig = item['text'].lower().strip()[:50]
            if sig not in seen_texts:
                valid_graph_results.append(item)
                seen_texts.add(sig)

        # 2. Tentukan berapa slot tersisa untuk Vector
        # Jika Graph memberikan hasil yang banyak, kita tetap sisakan ruang untuk Vector
        # Jika Graph kosong, Vector ambil alih semua slot (Fallback Mechanism)
        num_graph_used = len(valid_graph_results)
        
        # Strategi: Minimal 2 slot untuk Vector (agar konteks naratif tetap ada)
        target_vector_count = max(2, top_k - num_graph_used)
        
        # Jika Graph gagal total, Vector ambil semua slot (top_k)
        if num_graph_used == 0:
            target_vector_count = top_k

        # 3. Ambil dari Vector
        vec_results = self.vector_retriever.retrieve(query, top_k=target_vector_count + 2) # Ambil lebih untuk cadangan
        
        valid_vector_results = []
        for item in vec_results:
            sig = item['text'].lower().strip()[:50]
            if sig not in seen_texts:
                valid_vector_results.append(item)
                seen_texts.add(sig)

        # 4. Gabungkan: Graph (Reasoning) + Vector (Context)
        # Urutan: Graph ditaruh di atas agar LLM membacanya sebagai "Fakta Kunci"
        combined_results = valid_graph_results[:top_k] + valid_vector_results
        
        # Potong sesuai limit top_k akhir
        return combined_results[:top_k]