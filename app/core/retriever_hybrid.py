from app.core.retriever_vector import VectorRetriever
from app.core.retriever_graph import GraphRetriever

class HybridRetriever:
    def __init__(self):
        self.vector_retriever = VectorRetriever()
        self.graph_retriever = GraphRetriever()
    
    def retrieve(self, query: str, top_k: int = 5, alpha: float = 0.5):
        """
        Hybrid retrieval: Menggabungkan hasil graf dan vektor.
        alpha: Bobot untuk vektor
        """
        # Ambil dari vector
        vec_results = self.vector_retriever.retrieve(query, top_k=top_k)
        
        # Ambil dari graph
        graph_results = self.graph_retriever.retrieve(query)
        
        # Gabungkan (reranking), prioritaskan graph result (faithfullness)
        combined_results = []
        seen_texts = set()
        
        # Masukkan hasil graph dulu
        for item in graph_results:
            if item['text'] not in seen_texts:
                combined_results.append(item)
                seen_texts.add(item['text'])
                
        # Masukkan hasil vector
        for item in vec_results:
            if item['text'] not in seen_texts:
                combined_results.append(item)
                seen_texts.add(item['text'])
        return combined_results[:top_k]