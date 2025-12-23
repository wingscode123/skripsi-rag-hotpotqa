import pickle
import os
from app.core.config import DATA_PROCESSED_DIR, METADATA_NAME

def inspect_metadata():
    path = os.path.join(DATA_PROCESSED_DIR, METADATA_NAME)
    print(f"🔍 Memeriksa file: {path}")
    
    if not os.path.exists(path):
        print("❌ File tidak ditemukan! Cek path di config.py")
        return

    with open(path, 'rb') as f:
        data = pickle.load(f)
        
    print(f"✅ Tipe Data: {type(data)}")
    
    if isinstance(data, list):
        print(f"✅ Jumlah Item: {len(data)}")
        if len(data) > 0:
            print("\n--- 🧐 CONTOH DATA PERTAMA (Item 0) ---")
            first_item = data[0]
            print(first_item)
            
            print("\n--- 🔑 KUNCI (KEYS) YANG TERSEDIA ---")
            if isinstance(first_item, dict):
                print(list(first_item.keys()))
                
                # Cek detail metadata jika ada
                if 'metadata' in first_item:
                    print(f"\nIsi 'metadata': {first_item['metadata']}")
            else:
                print("Item bukan dictionary.")
        else:
            print("❌ List kosong.")
    else:
        print("❌ Data bukan list.")

if __name__ == "__main__":
    inspect_metadata()