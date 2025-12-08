import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from app.core.config import DATA_PROCESSED_DIR, EMBEDDING_MODEL, DEVICE

class VectorStore:
    def __init__(self, index_name="hotpot_vector_index"):
        self.index_path = os.path.join(DATA_PROCESSED_DIR, f"{index_name}.faiss")
        self.metadata_path = os.path.join(DATA_PROCESSED_DIR, f"{index_name}_meta.pkl")
        
        # Inisialisasi model embedding
        print(f"[VectorStore] Memuat model embedding: {EMBEDDING_MODEL} di {DEVICE}")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        
        self.index = None
        self.metadata = {}
        
    def create_index(self, documents: List[Dict[str, Any]] ,batch_size=32):
        """
        Membuat indeks vektor dari list dokumen (chunks)
        documents: List of dict {'id':str, 'text': str, ...}
        """
        texts = [doc['text'] for doc in documents]
        ids = [doc['id'] for doc in documents]
        
        print(f"[VectorStore] Memulai encoding {len(texts)} dokumen...")
        
        # Encode teks menjadi vektor
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True # Untuk cosine similarity            
        )
        dimension = embeddings.shape[1] # Biasanya 384 untuk all-miniLM
        print(f"[VectorStore] Dimensi Vektor: {dimension}")
        
        # Inisialisasi Indeks FAISS (IndexFlatIP)
        self.index = faiss.IndexFlatIP(dimension)
        
        # Pindahkan indeks ke GPU agar pencarian lebih cepat
        try:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print("[VectorStore] Menggunakan FAISS GPU Index")
        except Exception as e:
            print(f"[VectorStore] Warning: Gagal menggunakan GPU untuk indeks, fallback ke CPU. Error: {e}")
        
        # Masukkan vector ke indeks
        self.index.add(embeddings)
        
        # Simpan metadata (mapping index int faiss ke id dan teks)
        # FAISS menggunakan integer id, perlu map ke id string
        for i, doc in enumerate(documents):
            self.metadata[i] = doc
        print(f"[VectorStore] Berhasil mengindeks {self.index.ntotal} dokumen")
    
    def save(self):
        """
        Menyimpan   indeks FAISS dan metadata ke disk
        """
        if self.index is None:
            print("[VectorStore] Indeks kosong, tidak ada yang disimpan.")
            return

        print(f"[VectorStore] Mempersiapkan penyimpanan ke {self.index_path}...")
        # FAISS GPU index perlu dikonvert ke cpu sebelum disimpan ke disk
        # agar kompatibel ketika di-load
        try:
            index_cpu = faiss.index_gpu_to_cpu(self.index)
            print("[VectorStore] Indeks berhasil disalin dari GPU ke CPU RAM")
        except Exception as e:
            print(f"[VectorStore] Info: Tidak perlu dikonversi GPU-ke-CPU atau konversi gagal. Menggunakan indeks asli. Detail: {e}")
            index_cpu = self.index
        
        # Simpan Indeks ke FAISS
        try:
            faiss.write_index(index_cpu, self.index_path)
            if os.path.exists(self.index_path):
                print(f"[VectorStore] Sukses menulis file: {self.index_path}")
            else:
                print(f"[VectorStore] Gagal: File .faiss tidak muncul setelah fungsi write.")
        except Exception as e:
            print(f"[VectorStore] Critical Error: Error saat write_index: {e}")
        
        # Simpan metadata
        try:
            print(f"[VectorStore] Menyimpan metadata ke {self.metadata_path}...")
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            print(f"[VectorStore] Metadata tersimpan")
        except Exception as e:
            print(f"[VectorStore] Error menyimpan metadata: {e}")
            
    def load(self):
        """
        Memuat indeks dan metadata dari disk
        """
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            raise FileNotFoundError("File indeks tidak ditemukan. Jalankan proses indexing dulu.")
        print(f"[VectorStore] Memuat indeks dari {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        
        # Pindahkan ke gpu lagi setelah load
        try:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        except:
            pass
        print(f"[VectorStore] Memuat metadata dari {self.metadata_path}...")
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
    
    def search(self, query: str, top_k=5):
        """
        Melakukan pencarian semantik
        """
        # Encode query
        query_vec = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        
        # Search di FAISS
        # D = distances (skor), I = Indices (ID integer)
        D, I = self.index.search(query_vec, top_k)
        
        results = []
        for rank, (score, idx) in enumerate(zip(D[0], I[0])):
            if idx in self.metadata:
                doc = self.metadata[idx]
                results.append({
                    "score" : float(score),
                    "id" : doc['id'],
                    "text" : doc['text'],
                    "title" : doc['title'],
                    "metadata" : doc['metadata']
                })
        return results
        
    
    
            