import json
import os
from app.core.config import DATA_RAW_DIR

class HotpotQALoader:
    def __init__(self, file_name="hotpot_train_v1.1.json"):
        self.file_path = os.path.join(DATA_RAW_DIR, file_name)
    
    def load_data(self, limit=None):
        """
        Memuat data HotpotQA dari file json
        """
        print(f"[Loader] Memuat data dari: {self.file_path}")
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File tidak ditemukan di {self.file_path}")
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # HotpotQA training data adalah list of dict
            total_data = len(data)
            print(f"[Loader] Total data mentah ditemukan: {total_data}")
            
            if limit:
                data = data[:limit]
                print(f"[Loader] Mode Testing: Hanya mengambil {limit} data awal.")
            
            return data
        except Exception as e:
            print(f"[Loader] Error saat membaca file: {e}")
            return []
        
    def get_contexts(self, record):
        """
        Mengambil semua artikel konteks dari satu record.
        Hotpot QA context format: [[title], [sent1, sent2,...], ...]
        Output: List of dictionary {'title': str, 'sentences': list}
        """
        contexts = []
        raw_contexts = record.get('context', [])
        
        for item in raw_contexts:
            title = item[0]
            sentences = item[1]
            contexts.append({
                'title' : title,
                'sentences' : sentences
            })
        return contexts