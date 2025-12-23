import os
import json
import pickle
import pandas as pd
from tqdm import tqdm
from app.core.config import (
    DATA_RAW_DIR, 
    DATA_PROCESSED_DIR, 
    METADATA_NAME
)

def generate_closed_set_questions():
    print("=== 🎯 MENYIAPKAN DATA UJI (STRATEGI CLOSED-SET) [REVISI] ===")
    
    # 1. Tentukan Path File
    path_meta_pkl = os.path.join(DATA_PROCESSED_DIR, METADATA_NAME)
    path_raw_json = os.path.join(DATA_RAW_DIR, "hotpot_train_v1.1.json")
    output_csv = os.path.join(DATA_PROCESSED_DIR, "closed_set_test_questions.csv")
    
    print(f"📂 Membaca Metadata dari: {path_meta_pkl}")
    
    # 2. Load Metadata (.pkl)
    if not os.path.exists(path_meta_pkl):
        print(f"❌ Error: File metadata tidak ditemukan di {path_meta_pkl}")
        return

    with open(path_meta_pkl, 'rb') as f:
        chunks_data = pickle.load(f)
    
    # --- PERBAIKAN LOGIKA EKSTRAKSI ID ---
    print("🔍 Sedang mengekstrak ID unik dari metadata...")
    valid_ids = set()
    
    for chunk in tqdm(chunks_data, desc="Scanning Chunks"):
        # Kita ambil dari metadata -> original_question_id
        # Sesuai hasil inspeksi: {'metadata': {'original_question_id': '...'}}
        if 'metadata' in chunk and 'original_question_id' in chunk['metadata']:
             valid_ids.add(chunk['metadata']['original_question_id'])
            
    print(f"✅ Ditemukan {len(valid_ids)} ID unik Pertanyaan yang ter-index.")

    # 3. Load Raw JSON
    print(f"📂 Membaca Raw Data dari: {path_raw_json}")
    with open(path_raw_json, 'r') as f:
        raw_data = json.load(f)
        
    # 4. Filtering (Match ID Raw dengan ID Valid)
    print("🚀 Mencocokkan dengan Dataset Asli...")
    closed_set_questions = []
    
    for item in tqdm(raw_data, desc="Filtering Questions"):
        # HotpotQA raw menggunakan key '_id'
        if item['_id'] in valid_ids:
            closed_set_questions.append({
                'id': item['_id'],
                'question': item['question'],
                'answer': item['answer'],
                'type': item.get('type', 'unknown'),
                'level': item.get('level', 'unknown')
            })
            
    # 5. Simpan Hasil
    if closed_set_questions:
        df = pd.DataFrame(closed_set_questions)
        df.to_csv(output_csv, index=False)
        print(f"\n🎉 SUKSES! File soal ujian tersimpan di:")
        print(f"   -> {output_csv}")
        print(f"   Total Pertanyaan Siap Uji: {len(df)}")
        print("   Gunakan file ini untuk Dashboard dan Evaluasi Bab 6.")
    else:
        print("❌ Warning: Masih tidak ada pertanyaan yang cocok. Ada yang aneh.")

if __name__ == "__main__":
    generate_closed_set_questions()