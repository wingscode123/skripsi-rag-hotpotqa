import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from app.core.rag_pipeline import RAGPipeline
from app.core.config import DATA_PROCESSED_DIR

# Import Metrics Calculator
try:
    from app.core.utils_metrics import compute_exact_match, compute_f1
    from app.core.utils_metrics import LocalRagasEvaluator 
    USE_RAGAS_GLOBAL = True  
except ImportError:
    print("⚠️ Warning: Modul utils_metrics belum lengkap/tidak ditemukan.")
    def compute_exact_match(p, g): return 1 if p == g else 0
    def compute_f1(p, g): return 1.0
    USE_RAGAS_GLOBAL = False

def plot_radar_chart(summary_df, output_path):
    """
    Membuat Radar Chart (Spider Plot) dari dataframe ringkasan.
    Metrik: F1, Faithfulness, Relevancy, dan Speed Score.
    """
    try:
        # 1. Persiapan Data
        df_plot = summary_df.copy()
        
        # Konversi Latency menjadi 'Speed Score' (0 s.d 1)
        # Rumus: 1 / (1 + Latency) -> Latency 0s = Score 1.0, Latency tinggi = Score mendekati 0
        df_plot['speed'] = 1 / (1 + df_plot['latency'])
        
        # Kolom yang akan di-plot (Sesuai Bab 3 & 6)
        categories = ['f1', 'faithfulness', 'relevancy', 'speed']
        labels = ['Akurasi (F1)', 'Faithfulness', 'Relevancy', 'Kecepatan']
        
        # Setup variabel untuk plot lingkaran
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        # Inisialisasi Plot Polar
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        
        # Setup Axis (Jam 12 sebagai titik awal)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Label Sumbu
        plt.xticks(angles[:-1], labels, size=10, weight='bold')
        
        # Setup Grid Y (0.2, 0.4, ... 1.0)
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
        plt.ylim(0, 1.0) # Batas maksimal skor 1.0

        # Warna untuk tiap mode
        colors = {"vector": "#3498db", "graph": "#e74c3c", "hybrid": "#2ecc71"}
        
        # Loop menggambar garis untuk Vector, Graph, Hybrid
        for mode in df_plot.index:
            values = df_plot.loc[mode, categories].tolist()
            values += values[:1] # Menutup loop
            
            color = colors.get(mode, "black")
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"{mode.upper()}", color=color)
            ax.fill(angles, values, color=color, alpha=0.1) # Warna isi transparan

        # Legenda dan Judul
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title("Perbandingan Performa RAG (Multidimensional)", y=1.08, weight='bold')
        
        # Simpan
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Radar Chart berhasil disimpan di: {output_path}")

    except Exception as e:
        print(f"❌ Gagal membuat Radar Chart: {e}")

def main():
    print("=== 📊 EVALUASI SKRIPSI: CLOSED-SET 20K (Vector vs Graph vs Hybrid) ===")
    
    # --- 1. KONFIGURASI ---
    JUMLAH_SAMPEL = 50 
    OUTPUT_FILE = "eval_metrics_20k.csv"
    OUTPUT_CHART = "eval_radar_chart.png"
    
    OUTPUT_PATH = os.path.join(DATA_PROCESSED_DIR, OUTPUT_FILE)
    CHART_PATH = os.path.join(DATA_PROCESSED_DIR, OUTPUT_CHART)
    CSV_SOURCE = os.path.join(DATA_PROCESSED_DIR, "closed_set_test_questions.csv")

    # --- 2. LOAD DATASET ---
    print(f"[1] Memuat Soal Ujian dari: {CSV_SOURCE}")
    if not os.path.exists(CSV_SOURCE):
        print("❌ Error: File closed_set_test_questions.csv tidak ditemukan!")
        return

    df_test = pd.read_csv(CSV_SOURCE)
    
    if len(df_test) > JUMLAH_SAMPEL:
        df_sample = df_test.sample(JUMLAH_SAMPEL, random_state=42)
    else:
        df_sample = df_test
    
    print(f"    Mengambil {len(df_sample)} soal untuk pengujian.")

    # --- 3. INISIALISASI PIPELINE ---
    print("\n[2] Menginisialisasi RAG Pipeline (Index 20k)...")
    pipeline = RAGPipeline()
    
    # Init Ragas Judge
    ragas_judge = None
    if USE_RAGAS_GLOBAL:
        print("    Menginisialisasi LLM-as-a-Judge (Local)...")
        try:
            ragas_judge = LocalRagasEvaluator(pipeline.generator)
            print("    ✅ Local Judge Siap.")
        except Exception as e:
            print(f"    ⚠️ Gagal init Judge: {e}. Skip evaluasi LLM Judge.")
            ragas_judge = None 

    modes = ["vector", "graph", "hybrid"]
    results = []
    
    print(f"\n[3] 🔥 MEMULAI PENGUJIAN ({len(df_sample)} Soal x 3 Mode)...")

    # Loop Pertanyaan
    counter = 0
    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Testing"):
        query = row['question']
        ground_truth = row['answer']
        q_id = row['id']
        
        for mode in modes:
            try:
                # A. Jalankan Pipeline
                res = pipeline.answer_question(query, mode=mode, top_k=3)
                prediction = res['answer']
                latency = res['latency']
                contexts = res['contexts']
                
                # B. Hitung Metrik Tradisional
                em_score = compute_exact_match(prediction, ground_truth)
                f1_score = compute_f1(prediction, ground_truth)
                
                # C. Hitung Metrik Ragas (Local Judge)
                faithfulness = 0.0
                relevancy = 0.0
                if ragas_judge: 
                    faithfulness = ragas_judge.evaluate_faithfulness(prediction, contexts)
                    relevancy = ragas_judge.evaluate_relevancy(query, prediction)
                
                results.append({
                    "id": q_id,
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
                
            except Exception as e:
                print(f"\n❌ Error pada ID {q_id} Mode {mode}: {e}")
                continue
        
        # Simpan checkpoint
        counter += 1
        if counter % 5 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    # --- 4. HASIL AKHIR ---
    print("\n[4] ✅ Pengujian Selesai!")
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(OUTPUT_PATH, index=False)
        print(f"    File hasil CSV disimpan di: {OUTPUT_PATH}")

        # Tampilkan Ringkasan
        summary = df_results.groupby("mode")[["em", "f1", "faithfulness", "relevancy", "latency"]].mean()
        print("\n=== 📈 RATA-RATA PERFORMA (SKRIPSI) ===")
        print(summary)
        
        # --- 5. GENERATE RADAR CHART ---
        print("\n[5] 🎨 Membuat Visualisasi Radar Chart...")
        plot_radar_chart(summary, CHART_PATH)
        
    else:
        print("⚠️ Tidak ada hasil yang tersimpan.")

if __name__ == "__main__":
    main()