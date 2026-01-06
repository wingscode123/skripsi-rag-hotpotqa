import time
from typing import Dict, Any
from app.core.retriever_vector import VectorRetriever
from app.core.retriever_graph import GraphRetriever
from app.core.retriever_hybrid import HybridRetriever
from app.core.generator import LLMGenerator

class RAGPipeline:
    def __init__(self):
        """
        Inisialisasi semua komponen RAG (retriever & generator)
        """
        print("[Pipeline] Menginisialisasi komponen...")
        
        # Load generator
        self.generator = LLMGenerator()
        
        # Load retrievers
        self.vector_retriever = VectorRetriever()
        self.graph_retriever = GraphRetriever()
        self.hybrid_retriever = HybridRetriever()
        print("[Pipeline] Sistem siap.")
        
    def answer_question(self, query: str, mode: str = "hybrid", top_k: int = 5) -> Dict[str, Any]:
        """
        Fungsi utama untuk menjawab pertanyaan
        """
        start_time = time.time()
        
        # Retrieval
        print(f"[Pipeline] Mode: {mode.upper()} | Query: {query}")
        
        contexts = []
        if mode == "vector":
            contexts = self.vector_retriever.retrieve(query, top_k=top_k)
        elif mode == "graph":
            contexts = self.graph_retriever.retrieve(query)
            contexts = contexts[:top_k]
        elif mode == "hybrid":
            contexts = self.hybrid_retriever.retrieve(query, top_k=top_k)
        else: 
            return {"error" : "Mode tidak dikenal"}
        
        # Jika tidak ada konteks, beri tahu user
        if not contexts:
            return {
                "answer": "Sorry, the system could not find relevant information in the knowledge base to answer this question.",
                "contexts": [],
                "latency": time.time() -start_time
            }
        
        # Generation
        print(f"[Pipeline] Menghasilkan jawaban dari {len(contexts)} dokumen...")
        answer = self.generator.generate_answer(query, contexts)
        
        total_time = time.time() -start_time
        
        return {
            "question": query,
            "mode": mode,
            "answer": answer,
            "contexts": contexts,
            "latency": total_time
        }
        
        