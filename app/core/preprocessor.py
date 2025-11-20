import re
from typing import List, Dict

class TextPreprocessor:
    def clean_text(self, text: str) -> str:
        """
        Pembersihan teks meliputi:
        1. Menghapus spasi berlebih
        2. Normalisasi huruf (lowercase)
        3. Menghapus karakter aneh
        """
        if not text:
            return ""
        
        # Mengganti whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Lowercasing
        text = text.lower()
        return text.strip()
    
    def create_chunks(self, sentences: List[str], chunk_size=200, overlap=50) -> List[str]:
        """
        Menggabungkan kalimat menjadi chunk (passage) dengan ukuran tertentu.
        Strategi: Menggabungkan kalimat utuh hingga mendekati limit kata.
        """
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Gabungkan semua kalimat menjadi satu list agar mudah dihitung
        # Pendekatan berbasis kata, bukan token BPE, untuk kecepatan
        all_words = []
        for sent in sentences:
            # Bersihin setiap kalimat sebelum digabung
            clean_sent = self.clean_text(sent)
            all_words.extend(clean_sent.split())
        
        # Sliding window
        step = chunk_size - overlap
        for i in range(0, len(all_words), step):
            # Ambil slice setiap kata
            window = all_words[i : i + chunk_size]
            chunk_text = " ".join(window)
            
            # Filter chunk yang terlalu pendek
            if len(window) > 10:
                chunks.append(chunk_text)
        return chunks
    
    def process_record(self, record, chunk_size=200, overlap=50) -> List[Dict]:
        """
        Memproses satu record HotpotQA menjadi daftar chunks yang siap pakai
        """
        processed_docs = []
        question = self.clean_text(record['question'])
        record_id = record['_id']
        
        # Ambil konteks
        raw_contexts = record.get('context', [])
        
        for item in raw_contexts:
            title = item[0]
            sentences = item[1]
            
            # Melakukan chunking
            chunks = self.create_chunks(sentences, chunk_size, chunk_size-overlap)
            
            for idx, chunk_text in enumerate(chunks):
                # Struktur data untuk vector store dan graph linker
                doc = {
                    "id" : f"{record_id}_{title}_{idx}",
                    "title" : title,
                    "text" : chunk_text,
                    "metadata" : {
                        "source" : title,
                        "type" : "context",
                        "original_question_id" : record_id
                    }
                }
                processed_docs.append(doc)
        return processed_docs
    
    