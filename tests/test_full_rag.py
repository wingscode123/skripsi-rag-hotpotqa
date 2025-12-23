from app.core.rag_pipeline import RAGPipeline

def main():
    print("=== 🏁 PENGUJIAN FULL SYSTEM RAG ===")
    
    # 1. Inisialisasi Pipeline (Load semua model)
    pipeline = RAGPipeline()
    
    # 2. Pertanyaan Tes
    query = "What implies to the magnificent structure?" 
    
    # --- SKENARIO 1: VECTOR RAG ---
    print("\n" + "="*30)
    print("🧪 TEST 1: VECTOR RAG")
    print("="*30)
    result_v = pipeline.answer_question(query, mode="vector", top_k=3)
    print(f"⏱️ Latency: {result_v['latency']:.2f}s")
    print(f"📝 Jawaban:\n{result_v['answer']}")
    print("\n📚 Sumber yang dipakai:")
    for ctx in result_v['contexts']:
        print(f"- {ctx['title']} (Skor: {ctx.get('score', 0):.3f})")

    # --- SKENARIO 2: GRAPH RAG ---
    print("\n" + "="*30)
    print("🧪 TEST 2: GRAPH RAG")
    print("="*30)
    result_g = pipeline.answer_question(query, mode="graph")
    print(f"⏱️ Latency: {result_g['latency']:.2f}s")
    print(f"📝 Jawaban:\n{result_g['answer']}")
    
    # --- SKENARIO 3: HYBRID RAG ---
    print("\n" + "="*30)
    print("🧪 TEST 3: HYBRID RAG")
    print("="*30)
    result_h = pipeline.answer_question(query, mode="hybrid", top_k=3)
    print(f"⏱️ Latency: {result_h['latency']:.2f}s")
    print(f"📝 Jawaban:\n{result_h['answer']}")

if __name__ == "__main__":
    main()