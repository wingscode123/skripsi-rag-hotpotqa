import sys
import os
import streamlit as st
import pandas as pd
import random

# --- SETUP PATH ---
# Ambil direktori tempat dashboard.py berada (Root Project)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app.core.rag_pipeline import RAGPipeline
from app.core.config import DATA_PROCESSED_DIR

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Skripsi RAG: HotpotQA (20k Closed-Set)",
    page_icon="🤖",
    layout="wide"
)

# CSS Custom
st.markdown("""
    <style>
    .stChatMessage { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .source-box { font-size: 0.85rem; color: #555; background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-top: 5px; }
    .stButton button { width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI UTAMA ---

@st.cache_resource
def load_pipeline():
    return RAGPipeline()

@st.cache_data
def load_test_questions():
    # Load file CSV yang baru saja Anda buat
    csv_path = os.path.join(DATA_PROCESSED_DIR, "closed_set_test_questions.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        return None

# --- STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    rag_mode = st.selectbox(
        "Metode RAG:",
        ("hybrid", "vector", "graph"),
        index=0,
        format_func=lambda x: x.upper() + "-RAG"
    )
    top_k = st.slider("Top-K Context:", 1, 10, 3)
    
    st.divider()
    
    # FITUR BARU: LOAD PERTANYAAN DARI CSV
    st.subheader("🧪 Uji Closed-Set (20k)")
    df_questions = load_test_questions()
    
    if df_questions is not None:
        st.success(f"Terhubung: {len(df_questions)} Soal")
        if st.button("🎲 Ambil Soal Acak"):
            # Ambil 1 baris acak
            random_row = df_questions.sample(1).iloc[0]
            # Simpan ke session state agar masuk ke input box
            st.session_state.input_text = random_row['question']
            # Tampilkan contekan jawaban asli (Opsional, untuk debug)
            st.info(f"**Jawaban Asli (Gold):** {random_row['answer']}")
    else:
        st.warning("File 'closed_set_test_questions.csv' belum ditemukan.")

    st.divider()
    if st.button("🗑️ Bersihkan Chat"):
        st.session_state.messages = []
        st.session_state.input_text = ""
        st.rerun()

# --- MAIN PAGE ---
st.title("🤖 Demo Skripsi: Hybrid RAG (20k Data)")
st.caption("Eksperimen Closed-Set pada Dataset HotpotQA")

# Load Pipeline
with st.spinner("Memuat Index 20k & Model..."):
    pipeline = load_pipeline()

# Tampilkan Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "contexts" in message:
            with st.expander("📚 Lihat Bukti Dokumen"):
                for ctx in message["contexts"]:
                    score = f"(Score: {ctx.get('score', 0):.3f})"
                    st.markdown(f"**[{ctx.get('title', 'Doc')}]** {score}")
                    st.caption(ctx.get('text', '')[:300] + "...")
                    if 'triplets' in ctx:
                        st.json(ctx['triplets'])
                    st.divider()

# Input Area
user_input = st.chat_input("Ketik pertanyaan atau gunakan tombol dadu di sidebar...", key="chat_input")

# Handle Input dari Tombol Sidebar (Override)
if st.session_state.input_text and not user_input:
    # Jika ada input dari tombol dadu, kita paksa jadi user_input
    # Sayangnya st.chat_input agak tricky untuk di-set value-nya secara programmatik langsung trigger.
    # Jadi kita tampilkan instruksi:
    st.info(f"👇 **Pertanyaan Terpilih:**\n\n{st.session_state.input_text}\n\n*Silakan Copy-Paste ke kolom chat di bawah jika belum otomatis masuk.*")

# Logika Chat
if user_input:
    # Reset input text session
    st.session_state.input_text = ""
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner(f"Sedang berpikir ({rag_mode.upper()})..."):
            try:
                # EKSEKUSI PIPELINE
                result = pipeline.answer_question(query=user_input, mode=rag_mode, top_k=top_k)
                
                answer_text = result['answer']
                latency = result['latency']
                contexts = result['contexts']
                
                st.markdown(answer_text)
                st.caption(f"⏱️ Latency: {latency:.2f}s | Mode: {rag_mode.upper()}")
                
                with st.expander("📚 Lihat Bukti Dokumen"):
                    for ctx in contexts:
                        st.markdown(f"**[{ctx.get('title', 'Doc')}]** (Score: {ctx.get('score',0):.3f})")
                        st.text(ctx.get('text',''))
                        st.divider()
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "contexts": contexts
                })
                
            except Exception as e:
                st.error(f"Error: {e}")