import os
import sys
from typing import List, Dict
from llama_cpp import Llama
from app.core.config import MODEL_DIR, LLM_MODEL_FILE, DEVICE

class LLMGenerator:
    def __init__(self):
        """
        Inisialisasi model LLM (Mistral GGUF) secara lokal 
        """
        self.model_path = os.path.join(MODEL_DIR, LLM_MODEL_FILE)
        
        if not os.path.exists(self.model_path):
            print(f"[Generator] Error: Model tidak ditemukan di {self.model_path}")
            sys.exit(1)
        
        print(f"[Generator] Memuat model: {LLM_MODEL_FILE}...")
        
        # Konfigurasi GPU offload
        n_gpu_layers = -1 if DEVICE == "cuda" else 0
        
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=2048,
                verbose=False
            )
            print("[Generator] Model berhasil dimuat ke memori.")
        except Exception as e:
            print(f"[Generator] Gagal memuat model: {e}")
            sys.exit(1)
    
    def _format_prompt(self, query: str, contexts: List[Dict]) -> str:
        """
        Menyusun prompt sesuai format instruksi Mistral
        Format: <s>[INST] Instruksi + Konteks + Pertanyaan [/INST]
        """
        # Gabungkan teks konteks menjadi satu string
        context_str = ""
        for idx, item in enumerate(contexts):
            # Ambil text dari hasil retrieval
            text = item.get('text', '').strip()
            context_str += f"\n[Dokumen {idx+1}]: {text}"
        
        # Template prompt yang ketat agar model fokus pada konteks
        system_prompt = (
            "You are an intelligent research assistant. "
            "Answer the user's question ONLY based on the Facts/Context provided below. "
            "If the answer is not found in the context, say 'Sorry, the information is not available in the provided documents'. "
            "Do not fabricate information."
        )
        
        user_prompt = f"""
        === KONTEKS ====
        {context_str}
        
        === PERTANYAAN ===
        {query}
        
        === JAWABAN ===
        """
        
        # Format final mistral
        full_prompt = f"[INST] {system_prompt}\n{user_prompt} [/INST]"
        return full_prompt
    
    def generate_answer(self, query: str, contexts: List[Dict], max_tokens=256, temperature=0.1) -> str:
        """
        Menghasilkan jawaban dari LLM
        """
        if not contexts:
            return "Maaf, tidak ada konteks yang ditemukan untuk menjawab pertanyaan ini."
        
        prompt = self._format_prompt(query, contexts)
        
        # Streaming output, melakukan inferensi
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            stop=["</s>", "[/INST]"],
            temperature=temperature,
            top_p=0.95,
            echo=False
        )
        
        # Ambil teks jawaban
        answer = output['choices'][0]['text'].strip()
        return answer