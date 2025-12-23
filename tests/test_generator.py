import time
from app.core.generator import LLMGenerator

def main():
    print("=== 🧪 PENGUJIAN GENERATOR (LLM) ===")
    
    # 1. Inisialisasi
    start_load = time.time()
    generator = LLMGenerator()
    print(f"Waktu Loading Model: {time.time() - start_load:.2f} detik")

    # 2. Siapkan Data Dummy (Simulasi hasil Retrieval)
    dummy_query = "Who is the founder of Microsoft?"
    dummy_contexts = [
        {"text": "Bill Gates and Paul Allen founded Microsoft in 1975."},
        {"text": "Microsoft is a multinational technology corporation headquartered in Redmond."},
        {"text": "Steve Jobs was the co-founder of Apple Inc."} # Konteks pengecoh
    ]

    print(f"\n[Input] Pertanyaan: {dummy_query}")
    print(f"[Input] Jumlah Konteks: {len(dummy_contexts)}")

    # 3. Generate Jawaban
    print("\n[Proses] Sedang berpikir...")
    start_gen = time.time()
    
    answer = generator.generate_answer(dummy_query, dummy_contexts)
    
    print(f"\n[Output] Jawaban Model:\n{answer}")
    print(f"\n[Info] Waktu Generasi: {time.time() - start_gen:.2f} detik")

    print("\n=== PENGUJIAN SELESAI ===")

if __name__ == "__main__":
    main()