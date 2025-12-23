import pickle
import os
import networkx as nx
from app.core.config import DATA_PROCESSED_DIR

def main():
    path = os.path.join(DATA_PROCESSED_DIR, "knowledge_graph.pkl")
    
    if not os.path.exists(path):
        print("File graf tidak ditemukan.")
        return

    print("Loading Graph...")
    with open(path, "rb") as f:
        G = pickle.load(f)
    
    print(f"Total Nodes: {G.number_of_nodes()}")
    
    # Cari node yang mengandung kata "Arthur" atau "First"
    search_terms = ["arthur", "first", "women", "magazine"]
    
    print("\n=== HASIL PENCARIAN NODE ===")
    found = False
    for node in G.nodes():
        node_str = str(node).lower()
        for term in search_terms:
            if term in node_str:
                print(f"- '{node}'")
                found = True
                break
    
    if not found:
        print("Tidak ada node yang cocok dengan kata kunci.")

if __name__ == "__main__":
    main()