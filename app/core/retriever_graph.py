import pickle
import os
import networkx as nx
from app.core.config import DATA_PROCESSED_DIR

class GraphRetriever:
    def __init__(self, graph_file="knowledge_graph.pkl"):
        self.graph_path = os.path.join(DATA_PROCESSED_DIR, graph_file)
        self.G = None
        self.load_graph()
        
    def load_graph(self):
        if os.path.exists(self.graph_path):
            with open(self.graph_path, "rb") as f:
                self.G = pickle.load(f)
            print(f"[GraphRetriever] Graf dimuat: {self.G.number_of_nodes()} nodes.")
        else:
            print("[GraphRetriever] File graf tidak ditemukan.")
    
    def retrieve(self, query: str, depth: int = 1) -> list:
        """
        Melakukan graph transversal sederhana.
        1. cari entitas dalam query yang cocok dengan node di graf
        2. ambil neighbors dari node tersebut.
        """
        if not self.G:
            return []
        
        # Entity linking sederhana (string matching)
        start_nodes = []
        query_lower = query.lower()
        
        for node in self.G.nodes():
            if str(node).lower() in query_lower:
                start_nodes.append(node)
        print(f"[GraphRetriever] Entitas ditemukan di query: {start_nodes}")
        
        # Transversal (ambil konteks)
        retrieved_info = []
        for node in start_nodes:
            # ambil tetangga (1-hop)
            try:
                neighbors = list(self.G.neighbors(node))
                for neighbor in neighbors:
                    edge_data = self.G.get_edge_data(node, neighbor)
                    relation = edge_data.get('relation', 'related_to')
                    
                    # format menjadi teks kalimat (verbalisasi triplet)
                    fact = f"{node} {relation} {neighbor}"
                    retrieved_info.append({
                        "text" : fact,
                        "source_id" : edge_data.get('source_id'),
                        "score" : 1.0,
                        "type" : "graph_path"
                    })
            except Exception as e:
                continue
        return retrieved_info