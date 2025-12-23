# test_ingest.py
from app.core.data_loader import HotpotQALoader
from app.core.preprocessor import TextPreprocessor
import time

def main():
    print("=== MULAI PENGUJIAN DATA PIPELINE ===")
    
    # 1. Test Loader
    loader = HotpotQALoader(file_name="hotpot_train_v1.1.json")
    
    start_time = time.time()
    data = loader.load_data(limit=5)
    print(f"Waktu load (5 record): {time.time() - start_time:.4f} detik")
    
    if not data:
        print("❌ Gagal memuat data.")
        return

    # 2. Test Preprocessor
    processor = TextPreprocessor()
    
    # Ambil sampel pertama
    sample_record = data[0]
    print(f"\n[Sampel] Pertanyaan Asli: {sample_record['question']}")
    
    # Proses
    print("\n--- Memproses Context menjadi Chunks ---")
    chunks = processor.process_record(sample_record)
    
    print(f"Total chunks dihasilkan dari 1 pertanyaan: {len(chunks)}")
    
    # Tampilkan 2 chunk pertama
    for i, chunk in enumerate(chunks[:2]):
        print(f"\nChunk #{i+1}")
        print(f"ID   : {chunk['id']}")
        print(f"Title: {chunk['title']}")
        print(f"Text : {chunk['text'][:150]}...") # Tampilkan 150 karakter awal saja

    print("\n=== PENGUJIAN SELESAI ===")

if __name__ == "__main__":
    main()