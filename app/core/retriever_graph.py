import pickle
import os
import networkx as nx
from app.core.config import DATA_PROCESSED_DIR

class GraphRetriever:
    def __init__(self, graph_file="knowledge_graph_20k.pkl"):
        self.graph_path = os.path.join(DATA_PROCESSED_DIR, graph_file)
        self.G = None
        self.load_graph()

    def load_graph(self):
        if os.path.exists(self.graph_path):
            with open(self.graph_path, "rb") as f:
                self.G = pickle.load(f)
            print(f"✅ [GraphRetriever] GRAF DIMUAT: {self.G.number_of_nodes()} NODES.")
        else:
            print("❌ [GraphRetriever] File graf tidak ditemukan.")

    def retrieve(self, query: str, depth: int = 1) -> list:
        if not self.G:
            return []

        print(f"🔍 [GraphRetriever] Memproses Query: '{query}'")
        
        # 1. Preprocessing & Entity Linking
        query_clean = query.lower().replace("?", "").replace(".", "").replace(",", "")
        query_tokens = set(query_clean.split())
        stopwords = {"what", "who", "is", "the", "a", "of", "in", "tell", "me", "about"}
        keywords = query_tokens - stopwords
        
        start_nodes = []
        for node in self.G.nodes():
            node_str = str(node).lower().strip()
            if len(node_str) < 3: continue
            
            # Exact match
            if node_str in query_clean:
                start_nodes.append(node)
                continue
            # Keyword match
            node_tokens = set(node_str.split())
            if len(node_tokens.intersection(keywords)) >= 1:
                 if len(node_tokens) > 2 and len(node_tokens.intersection(keywords)) < 2:
                     continue
                 start_nodes.append(node)

        start_nodes.sort(key=len, reverse=True)
        start_nodes = start_nodes[:5] # Ambil 5 entitas terkuat
        print(f"   [Info] Entitas kandidat: {start_nodes}")

        # 2. Traversal Dua Arah (Incoming & Outgoing)
        retrieved_info = []
        seen_facts = set()

        for node in start_nodes:
            # A. Cek Outgoing (Node -> Tetangga)
            try:
                successors = list(self.G.successors(node))
                for neighbor in successors:
                    edge_data = self.G.get_edge_data(node, neighbor)
                    relation = edge_data.get('relation', 'related_to')
                    fact = f"{node} ({relation}) {neighbor}"
                    if fact not in seen_facts:
                        retrieved_info.append({
                            "text": fact, 
                            "chunk_id": edge_data.get('source_id'), 
                            "score": 1.0, "type": "graph_out", "title": f"Graph: {node}"
                        })
                        seen_facts.add(fact)
            except: pass

            # B. Cek Incoming (Tetangga -> Node)
            try:
                predecessors = list(self.G.predecessors(node))
                for neighbor in predecessors:
                    # Perhatikan urutan: neighbor -> node
                    edge_data = self.G.get_edge_data(neighbor, node) 
                    relation = edge_data.get('relation', 'related_to')
                    fact = f"{neighbor} ({relation}) {node}" # Kalimat dibalik agar logis
                    if fact not in seen_facts:
                        retrieved_info.append({
                            "text": fact, 
                            "source_id": edge_data.get('source_id'), 
                            "score": 1.0, "type": "graph_in", "title": f"Graph: {node}"
                        })
                        seen_facts.add(fact)
            except: pass
        return retrieved_info[:20]