import time
import argparse
import os
import pickle
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from app.core.data_loader import HotpotQALoader
from app.core.preprocessor import TextPreprocessor
from app.core.vector_store import VectorStore
from app.core.extractor import TripletExtractor
from app.core.config import DATA_PROCESSED_DIR

# --- CLASS LOGGER ---
class IndexingLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.stats = {}
        with open(self.log_file, "w") as f:
            f.write(f"=== LOG INDEXING STARTED AT {datetime.now()} ===\n")

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {message}"
        print(full_msg)
        with open(self.log_file, "a") as f:
            f.write(full_msg + "\n")

    def start_timer(self, stage_name):
        self.stats[stage_name] = {'start': time.time()}
        self.log(f"🚀 MEMULAI TAHAP: {stage_name}")

    def end_timer(self, stage_name, extra_info=""):
        if stage_name in self.stats:
            duration = time.time() - self.stats[stage_name]['start']
            self.stats[stage_name]['duration'] = duration
            self.log(f"✅ SELESAI TAHAP: {stage_name} | Durasi: {duration:.2f} detik {extra_info}")
            return duration
        return 0

    def get_duration(self, stage_name):
        return self.stats.get(stage_name, {}).get('duration', 0)

# --- FUNGSI VISUALISASI CHART ---
def plot_indexing_comparison(vector_time, graph_time, output_path):
    """
    Membuat Bar Chart perbandingan waktu indexing.
    Menggunakan skala Logaritmik agar perbedaan ekstrem tetap terlihat.
    """
    labels = ['Vector Indexing\n(FAISS)', 'Graph Construction\n(Extraction + Build)']
    times = [vector_time, graph_time]
    colors = ['#3498db', '#e74c3c'] # Biru untuk Vector, Merah untuk Graph

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, times, color=colors, alpha=0.8, width=0.5)

    # Set Log Scale karena perbedaan durasi biasanya ekstrem (detik vs jam)
    plt.yscale('log')
    plt.ylabel('Waktu Eksekusi (Detik) - Skala Log')
    plt.title('Perbandingan Indexing Cost: Vector vs Graph')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Menambahkan label angka di atas batang
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                 f'{height:.1f} s',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Tambahkan keterangan rasio di bawah grafik
    if vector_time > 0:
        ratio = graph_time / vector_time
        plt.figtext(0.5, 0.01, f"Catatan: Graph Construction {ratio:.1f}x lebih lambat dari Vector Indexing", 
                    ha="center", fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Chart perbandingan disimpan ke: {output_path}")

def build_networkx_graph(df_triplets):
    G = nx.DiGraph()
    for _, row in df_triplets.iterrows():
        G.add_edge(row['head'], row['tail'], relation=row['relation'], source_id=row['chunk_id'])
    return G

def main(args):
    # Setup Path & Nama File
    suffix = args.suffix
    vector_index_name = f"hotpot_{suffix}"
    triplet_csv_name = f"triplets_{suffix}.csv"
    checkpoint_name = f"triplets_checkpoint_{suffix}.csv"
    graph_pkl_name = f"knowledge_graph_{suffix}.pkl"
    linking_csv_name = f"entity_linking_{suffix}.csv"
    
    log_file_name = f"indexing_log_{suffix}.txt"
    chart_file_name = f"indexing_cost_chart_{suffix}.png" # Nama file gambar chart
    
    final_csv_path = os.path.join(DATA_PROCESSED_DIR, triplet_csv_name)
    checkpoint_csv_path = os.path.join(DATA_PROCESSED_DIR, checkpoint_name)
    graph_pkl_path = os.path.join(DATA_PROCESSED_DIR, graph_pkl_name)
    linking_csv_path = os.path.join(DATA_PROCESSED_DIR, linking_csv_name)
    log_path = os.path.join(DATA_PROCESSED_DIR, log_file_name)
    chart_path = os.path.join(DATA_PROCESSED_DIR, chart_file_name)

    # Init Logger
    logger = IndexingLogger(log_path)
    
    limit_text = "ALL" if args.limit == -1 else str(args.limit)
    logger.log(f"KONFIGURASI: Limit={limit_text}, Batch={args.batch_size}, Suffix='{suffix}'")
    
    start_global = time.time()

    # --- 1. Load & Preprocess ---
    logger.start_timer("Preprocessing")
    loader = HotpotQALoader(file_name="hotpot_train_v1.1.json")
    load_limit = None if args.limit == -1 else args.limit
    raw_data = loader.load_data(limit=load_limit)
    
    preprocessor = TextPreprocessor()
    all_chunks = []
    for record in raw_data:
        chunks = preprocessor.process_record(record)
        all_chunks.extend(chunks)
    
    total_chunks = len(all_chunks)
    logger.end_timer("Preprocessing", f"| Total Chunks: {total_chunks}")

    # --- 2. Vector Indexing ---
    logger.start_timer("Vector Indexing")
    vector_store = VectorStore(index_name=vector_index_name)
    vector_store.create_index(all_chunks, batch_size=args.batch_size * 2)
    vector_store.save()
    logger.end_timer("Vector Indexing", "| FAISS Index Created")

    # --- 3. Graph Extraction ---
    logger.start_timer("Graph Extraction")
    extractor = TripletExtractor()
    all_triplets = []
    start_index = 0 
    
    # Resume Logic
    if os.path.exists(checkpoint_csv_path):
        logger.log(f"⚠️ Checkpoint ditemukan: {checkpoint_name}")
        try:
            df_checkpoint = pd.read_csv(checkpoint_csv_path)
            if not df_checkpoint.empty:
                all_triplets = df_checkpoint.to_dict('records')
                last_chunk_id = df_checkpoint.iloc[-1]['chunk_id']
                found = False
                for idx in range(len(all_chunks)-1, -1, -1):
                    if all_chunks[idx]['chunk_id'] == last_chunk_id:
                        start_index = idx + 1
                        remainder = start_index % args.batch_size
                        if remainder != 0:
                            start_index = start_index + (args.batch_size - remainder)
                        found = True
                        break
                if found:
                    logger.log(f"⏩ RESUME dari Chunk ke-{start_index}...")
                else:
                    logger.log("⚠️ ID checkpoint tidak cocok. Mulai dari 0.")
        except Exception:
            logger.log("❌ Error baca checkpoint. Mulai dari 0.")

    # Loop Batch
    for i in range(start_index, total_chunks, args.batch_size):
        batch_chunks = all_chunks[i : i + args.batch_size]
        try:
            df_batch = extractor.process_batch(batch_chunks, batch_size=args.batch_size)
            if not df_batch.empty:
                all_triplets.extend(df_batch.to_dict('records'))
        except Exception as e:
            logger.log(f"❌ Error Batch {i}: {e}")
            continue
        
        current_count = i + args.batch_size
        if i % 1000 == 0:
            print(f"   ...Processing chunk {i}/{total_chunks}")
        if current_count % args.save_every < args.batch_size or current_count >= total_chunks:
            logger.log(f"💾 Checkpoint: {len(all_triplets)} triplets saved.")
            pd.DataFrame(all_triplets).to_csv(checkpoint_csv_path, index=False)

    logger.end_timer("Graph Extraction", f"| Total Triplet: {len(all_triplets)}")

    # --- 4. Build Graph Object ---
    logger.start_timer("Graph Construction")
    if not all_triplets:
        logger.log("⚠️ Graf Kosong! Tidak ada triplet.")
        return

    df_final = pd.DataFrame(all_triplets).drop_duplicates()
    df_final.to_csv(final_csv_path, index=False)
    G = build_networkx_graph(df_final)
    with open(graph_pkl_path, "wb") as f:
        pickle.dump(G, f)
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    logger.end_timer("Graph Construction", f"| Nodes: {num_nodes}, Edges: {num_edges}")

    # --- 5. Linking Table ---
    logger.start_timer("Entity Linking")
    linking_data = []
    for _, row in df_final.iterrows():
        linking_data.append({'entity': row['head'], 'chunk_id': row['chunk_id']})
        linking_data.append({'entity': row['tail'], 'chunk_id': row['chunk_id']})
    df_linking = pd.DataFrame(linking_data).drop_duplicates()
    df_linking.to_csv(linking_csv_path, index=False)
    logger.end_timer("Entity Linking")

    # --- FINISH & PLOTTING ---
    total_time = time.time() - start_global
    logger.log(f"=== ✅ INDEXING COMPLETE IN {total_time:.2f} SECONDS ===")
    
    # Ambil data durasi untuk visualisasi
    time_vector = logger.get_duration("Vector Indexing")
    time_extract = logger.get_duration("Graph Extraction")
    time_build = logger.get_duration("Graph Construction")
    total_graph_time = time_extract + time_build
    
    # Generate Chart Otomatis
    plot_indexing_comparison(time_vector, total_graph_time, chart_path)

    # Tulis Ringkasan
    with open(log_path, "a") as f:
        f.write("\n=== RINGKASAN UNTUK LAPORAN ===\n")
        f.write(f"Total Dokumen (Chunks): {total_chunks}\n")
        f.write(f"Vector Indexing Cost: {time_vector:.2f} s\n")
        f.write(f"Graph Construction Cost (Extract+Build): {total_graph_time:.2f} s\n")
        f.write(f"Statistik Graf Akhir: {num_nodes} Nodes, {num_edges} Edges\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--suffix", type=str, default="full")
    args = parser.parse_args()
    main(args)