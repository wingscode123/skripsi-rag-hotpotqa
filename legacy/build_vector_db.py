import time
from app.core.data_loader import HotpotQALoader
from app.core.preprocessor import TextPreprocessor
from app.core.vector_store import VectorStore

def main():
    print("=== 🚀 MEMULAI PROSES VECTOR INDEXING ===")
    start_global = time.time()

    # 1. Load Data
    LIMIT_DATA = 100  
    
    loader = HotpotQALoader(file_name="hotpot_train_v1.1.json")
    raw_data = loader.load_data(limit=LIMIT_DATA)
    
    if not raw_data:
        print("Data kosong!")
        return

    # 2. Preprocessing & Chunking
    print("\n[2] Melakukan Preprocessing & Chunking...")
    preprocessor = TextPreprocessor()
    all_documents = []
    
    for record in raw_data:
        chunks = preprocessor.process_record(record)
        all_documents.extend(chunks)
        
    print(f"Total dokumen (chunks) yang akan di-embed: {len(all_documents)}")
    
    # 3. Vector Store Creation
    print("\n[3] Membangun Vector Store (FAISS)...")
    vector_store = VectorStore(index_name="hotpot_v1")
    
    # Proses encoding dan indexing
    start_embed = time.time()
    vector_store.create_index(all_documents, batch_size=64) # Batch 64 cukup aman untuk RTX 3050 Ti
    print(f"Waktu Encoding & Indexing: {time.time() - start_embed:.2f} detik")
    
    # 4. Simpan
    print("\n[4] Menyimpan ke Disk...")
    vector_store.save()
    
    # 5. Test Search Sederhana
    print("\n[5] Pengujian Pencarian Cepat...")
    test_query = "What implies to the magnificent structure?" # Contoh query
    results = vector_store.search(test_query, top_k=3)
    
    print(f"Query: {test_query}")
    for res in results:
        print(f"- [{res['score']:.4f}] {res['title']}: {res['text'][:100]}...")

    print(f"\n=== SELESAI (Total Waktu: {time.time() - start_global:.2f} detik) ===")

if __name__ == "__main__":
    main()