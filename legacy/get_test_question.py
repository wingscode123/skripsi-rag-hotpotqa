import json
import random
import os
from app.core.config import DATA_RAW_DIR

def main():
    file_path = os.path.join(DATA_RAW_DIR, "hotpot_train_v1.1.json")
    
    print(f"Membaca file: {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    JUMLAH_DATA_INDEXED = 90000
    
    valid_data = data[:JUMLAH_DATA_INDEXED]
    
    print(f"Total Data Valid (Indexed): {len(valid_data)}")
    print("=== 5 CONTOH PERTANYAAN DARI DATABASE ANDA ===")
    print("Gunakan pertanyaan ini di Streamlit untuk menguji GraphRAG.\n")
    
    # Ambil 5 sampel acak
    samples = random.sample(valid_data, 5)
    
    for i, item in enumerate(samples):
        print(f"Pertanyaan {i+1}:")
        print(f"Q: {item['question']}")
        print(f"A: {item['answer']}")
        print(f"Tipe: {item['type']} | Level: {item['level']}")
        print("-" * 50)

if __name__ == "__main__":
    main()