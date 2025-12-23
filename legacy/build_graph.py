import time
import os
import pickle
import pandas as pd
import networkx as nx
from app.core.data_loader import HotpotQALoader
from app.core.preprocessor import TextPreprocessor
from app.core.extractor import TripletExtractor
from app.core.config import DATA_PROCESSED_DIR

def build_networkx_graph(df_triplets):
    """Konversi DataFrame triplet menjadi objek NetworkX"""
    G = nx.DiGraph() # Directed Graph (Berarah)
    
    print(f"[GraphBuilder] Membangun graf dari {len(df_triplets)} relasi...")
    for _, row in df_triplets.iterrows():
        head = row['head']
        tail = row['tail']
        relation = row['relation']
        G.add_edge(head, tail, relation=relation, source_id=row['chunk_id'])
        
    return G

def main():
    print("=== 🕸️ MEMULAI PROSES KONSTRUKSI GRAPH (GraphRAG) ===")
    start_global = time.time()

    # 1. Load Data
    LIMIT_DATA = 20 
    
    loader = HotpotQALoader(file_name="hotpot_train_v1.1.json")
    raw_data = loader.load_data(limit=LIMIT_DATA)
    
    if not raw_data:
        print("Data kosong atau tidak ditemukan.")
        return

    # 2. Preprocessing
    print("\n[2] Preprocessing & Chunking...")
    preprocessor = TextPreprocessor()
    all_chunks = []
    
    for record in raw_data:
        chunks = preprocessor.process_record(record)
        all_chunks.extend(chunks)
        
    print(f"Total chunks yang akan diproses: {len(all_chunks)}")
    
    # 3. Ekstraksi Relasi (REBEL)
    print("\n[3] Ekstraksi Triplet dengan REBEL...")
    print("⚠️  Note: Akan mengunduh model ~3GB pada run pertama.")
    
    extractor = TripletExtractor()
    start_extract = time.time()
    df_triplets = extractor.process_batch(all_chunks, batch_size=4)
    
    print(f"Waktu Ekstraksi: {time.time() - start_extract:.2f} detik")
    print(f"Total Triplet ditemukan: {len(df_triplets)}")
    
    if len(df_triplets) == 0:
        print("Tidak ada triplet yang terekstrak. Coba periksa data.")
        return

    # 4. Simpan Triplet Mentah (CSV)
    triplet_path = os.path.join(DATA_PROCESSED_DIR, "triplets_raw.csv")
    df_triplets.to_csv(triplet_path, index=False)
    print(f"[Save] Triplet mentah disimpan ke: {triplet_path}")

    # 5. Bangun Graf NetworkX & Simpan
    print("\n[4] Membangun Struktur Graf...")
    G = build_networkx_graph(df_triplets)
    
    print(f"Statistik Graf:")
    print(f"- Jumlah Node (Entitas): {G.number_of_nodes()}")
    print(f"- Jumlah Edge (Relasi) : {G.number_of_edges()}")
    
    graph_path = os.path.join(DATA_PROCESSED_DIR, "knowledge_graph.pkl")
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)
    print(f"[Save] Objek Graf disimpan ke: {graph_path}")
    
    # 6. Buat Entity Linking Table (Untuk Hybrid-RAG)
    print("\n[5] Membuat Entity-Text Linking Table...")
    linking_data = []
    
    for _, row in df_triplets.iterrows():
        linking_data.append({'entity': row['head'], 'chunk_id': row['chunk_id']})
        linking_data.append({'entity': row['tail'], 'chunk_id': row['chunk_id']})
        
    df_linking = pd.DataFrame(linking_data).drop_duplicates()
    link_path = os.path.join(DATA_PROCESSED_DIR, "entity_linking.csv")
    df_linking.to_csv(link_path, index=False)
    print(f"[Save] Linking table disimpan ke: {link_path}")

    print(f"\n=== SELESAI (Total Waktu: {time.time() - start_global:.2f} detik) ===")

if __name__ == "__main__":
    main()