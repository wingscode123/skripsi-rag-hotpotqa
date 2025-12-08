import time
import json
import random
import os
import pandas as pd
from tqdm import tqdm
from app.core.rag_pipeline import RAGPipeline
from app.core.config import DATA_RAW_DIR, DATA_PROCESSED_DIR
from app.core.utils_metrics import compute_exact_match, compute_f1, LocalRagasEvaluator

def main():
    print("=== 📊 EVALUASI SISTEM (EM, F1, Faithfulness, Relevancy) ===")
    
    # 1. Konfigurasi
    JUMLAH_SAMPEL = 20
    DATA_INDEXED = 2000
    
    # 2. Load Data Valid
    print("[1] Memuat Dataset...")
    file_path = os.path.join(DATA_RAW_DIR, "hotpot_train_v1.1.json")
    with open(file_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    valid_data = all_data[:DATA_INDEXED]
    test_samples = random.sample(valid_data, JUMLAH_SAMPEL)
    
    # 3. Inisialisasi Pipeline & Evaluator
    print("[2] Menginisialisasi Sistem...")
    pipeline = RAGPipeline()
    
    # Inisialisasi Juri (menggunakan model generator yang sama)
    ragas_judge = LocalRagasEvaluator(pipeline.generator)
    
    modes = ["vector", "graph", "hybrid"]
    results = []
    
    print(f"\n[3] Memulai Pengujian ({JUMLAH_SAMPEL} Soal x 3 Mode)...")
    
    for item in tqdm(test_samples, desc="Progress"):
        query = item['question']
        ground_truth = item['answer']
        
        for mode in modes:
            # A. Jalankan RAG
            res = pipeline.answer_question(query, mode=mode, top_k=5)
            prediction = res['answer']
            latency = res['latency']
            contexts = res['contexts']
            
            # B. Hitung Metrik Tradisional (EM & F1)
            em_score = compute_exact_match(prediction, ground_truth)
            f1_score = compute_f1(prediction, ground_truth)
            
            # C. Hitung Metrik RAGAS (LLM-as-a-Judge)
            faithfulness = ragas_judge.evaluate_faithfulness(prediction, contexts)
            relevancy = ragas_judge.evaluate_relevancy(query, prediction)
            
            results.append({
                "mode": mode,
                "em": em_score,
                "f1": f1_score,
                "faithfulness": faithfulness,
                "relevancy": relevancy,
                "latency": latency,
                "question": query,
                "prediction": prediction,
                "ground_truth": ground_truth
            })

    # 4. Simpan & Tampilkan
    print("\n[4] Hasil Evaluasi:")
    df = pd.DataFrame(results)
    output_path = os.path.join(DATA_PROCESSED_DIR, "eval_metrics_complete.csv")
    df.to_csv(output_path, index=False)
    print(f"    Disimpan ke: {output_path}")
    
    # Grouping Rata-rata
    summary = df.groupby("mode")[["em", "f1", "faithfulness", "relevancy", "latency"]].mean()
    print("\n=== RATA-RATA PERFORMA ===")
    print(summary)

if __name__ == "__main__":
    main()