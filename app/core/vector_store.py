import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from app.core.config import DATA_PROCESSED_DIR, METADATA_NAME

class VectorStore:
    def __init__(self, index_name="hotpot_20k"):
        self.index_name = index_name
        
        # Hardcode nama model (pastikan sama dengan saat indexing)
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.model_name)
        
        # Dimensi embedding
        self.dimension = 384 
        
        # Inisialisasi Index FAISS
        self.index = faiss.IndexFlatL2(self.dimension)
        
        self.chunks = [] 
        
        # --- LOGIKA PENENTUAN PATH METADATA ---
        if "20k" in index_name:
            self.metadata_path = os.path.join(DATA_PROCESSED_DIR, "hotpot_20k_meta.pkl")
        else:
            self.metadata_path = os.path.join(DATA_PROCESSED_DIR, f"{self.index_name}_meta.pkl")
            
        self.index_path = os.path.join(DATA_PROCESSED_DIR, f"{self.index_name}.faiss")

    def create_index(self, chunks, batch_size=32):
        self.chunks = chunks 
        total = len(chunks)
        print(f"[VectorStore] Memulai encoding {total} dokumen...")
        
        for i in range(0, total, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_texts = [c['text'] for c in batch_chunks]
            
            # Encode
            embeddings = self.model.encode(
                batch_texts, 
                batch_size=len(batch_texts),
                show_progress_bar=False, 
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
            self.index.add(embeddings)
            
            if i % 1000 == 0:
                print(f"Batches: {i}/{total} encoded...", end='\r')
        
        print(f"\n[VectorStore] Selesai. Total vectors: {self.index.ntotal}")

    def save(self):
        print(f"[VectorStore] Menyimpan index ke {self.index_path}...")
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        print("[VectorStore] Penyimpanan berhasil.")

    def load(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index not found: {self.index_path}")
        
        print(f"[VectorStore] Memuat index dari {self.index_path}...")
        self.index = faiss.read_index(self.index_path)
        
        print(f"[VectorStore] Memuat metadata dari {self.metadata_path}...")
        with open(self.metadata_path, 'rb') as f:
            self.chunks = pickle.load(f)
        print(f"[VectorStore] Berhasil memuat {len(self.chunks)} dokumen.")

    def search(self, query, top_k=5):
        # 1. Encode Query
        query_vector = self.model.encode([query], convert_to_numpy=True)
        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
        
        # 2. Search FAISS
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks) and idx >= 0:
                item = self.chunks[idx]
                
                # Menggunakan .get() agar aman. 
                # Prioritas: 'chunk_id' -> 'id' -> 'unknown'
                c_id = item.get('chunk_id', item.get('id', f'unknown_{idx}'))
                
                results.append({
                    "chunk_id": c_id,        
                    "text": item.get('text', ''),
                    "title": item.get('title', 'Untitled'),
                    "score": float(distances[0][i]),
                    "metadata": item.get('metadata', {})
                })
        return results