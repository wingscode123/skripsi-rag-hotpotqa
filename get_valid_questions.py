import json
import random
import os
from app.core.config import DATA_RAW_DIR

def main():
    # KONFIGURASI: Samakan dengan jumlah data yang Anda index di run_indexing.py
    JUMLAH_DATA_INDEXED = 2000 
    
    print(f"=== GENERATOR PERTANYAAN VALID (DARI {JUMLAH_DATA_INDEXED} DATA) ===\n")
    
    file_path = os.path.join(DATA_RAW_DIR, "hotpot_train_v1.1.json")
    
    if not os.path.exists(file_path):
        print("File dataset tidak ditemukan.")
        return

    print("Membaca file JSON (sedikit lama)...")
    with open(file_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    # PENTING: Kita memotong data persis seperti saat indexing
    # Jadi kita hanya mengambil kolam data yang "diketahui" oleh sistem
    known_data = all_data[:JUMLAH_DATA_INDEXED]
    
    print(f"Total Data yang diketahui sistem: {len(known_data)}")
    
    # Ambil 5 sampel acak dari kolam data yang valid ini
    samples = random.sample(known_data, 5)
    
    print("\nBerikut 5 pertanyaan yang JAWABANNYA PASTI ADA di sistem Anda:")
    print("="*60)
    
    for i, item in enumerate(samples):
        print(f"Pertanyaan #{i+1}")
        print(f"Q: {item['question']}")
        print(f"A: {item['answer']}")
        print(f"Tipe: {item['type']} | Level: {item['level']}")
        print("-" * 60)

if __name__ == "__main__":
    main()