from app.core.retriever_vector import VectorRetriever
from app.core.retriever_graph import GraphRetriever

class HybridRetriever:
    def __init__(self):
        # Inisialisasi dua jalur retriever terpisah (Vector & Graph)
        self.vector_retriever = VectorRetriever(index_name="hotpot_20k")
        self.graph_retriever = GraphRetriever(graph_file="knowledge_graph_20k.pkl")

    def retrieve(self, query: str, top_k: int = 5, alpha: float = 0.5):
        """
        Implementasi Hybrid-RAG dengan strategi Parallel Fusion.
        Menggabungkan penelusuran struktural (Graph) dan semantik (Vector)
        untuk melengkapi konteks multi-hop.
        """
        combined_results = []
        seen_texts = set()

        # 1. Structural Retrieval (Graph): Fokus menangkap relasi eksplisit antar entitas
        graph_results = self.graph_retriever.retrieve(query)
        
        # Filter duplikasi hasil graph
        valid_graph_results = []
        for item in graph_results:
            # Normalisasi signature untuk deduplikasi
            sig = item['text'].lower().strip()[:50]
            if sig not in seen_texts:
                valid_graph_results.append(item)
                seen_texts.add(sig)

        # Hitung sisa slot untuk Vector (Dynamic Filling)
        num_graph_used = len(valid_graph_results)
        target_vector_count = max(2, top_k - num_graph_used)
        
        if num_graph_used == 0:
            target_vector_count = top_k

        # 2. Semantic Retrieval (Vector): Melengkapi konteks yang tidak tertangkap Graph
        vec_results = self.vector_retriever.retrieve(query, top_k=target_vector_count + 2)
        
        valid_vector_results = []
        for item in vec_results:
            sig = item['text'].lower().strip()[:50]
            if sig not in seen_texts:
                valid_vector_results.append(item)
                seen_texts.add(sig)

        # Context Fusion: Menggabungkan hasil Graph (prioritas) dan Vector
        combined_results = valid_graph_results[:top_k] + valid_vector_results
        return combined_results[:top_k]