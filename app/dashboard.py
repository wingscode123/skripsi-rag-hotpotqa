import sys
import os

# Dapatkan path absolut file ini
current_file_path = os.path.abspath(__file__)
# Ambil path folder 'app'
app_dir = os.path.dirname(current_file_path)
# Dapatkan folder root 'skripsi_rag
root_dir = os.path.dirname(app_dir)

# Masukkan root_dir ke sys.path agar pyhon mengenali 'app' sebagai package
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import streamlit as st
from app.core.rag_pipeline import RAGPipeline


# Konfigurasi halaman
st.set_page_config(
    page_title="Skripsi RAG: HotpotQA",
    page_icon="🤖",
    layout="wide"
)

# CSS custom untuk tampilan
st.markdown(
    """
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .source-box {
        font-size: 0.85rem;
        color: #555;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Fungsi caching
@st.cache_resource
def load_pipeline():
    return RAGPipeline()

# Inisialisasi
if "messages" not in st.session_state:
    st.session_state.messages = []
    
st.title("Analisis Komparatif RAG (HotpotQA)")
st.markdown("Sistem Multi-hop QA dengan pendekatan Vector, Graph, dan Hybrid.")

# Load pipeline
with st.spinner("Memuat model & index...."):
    pipeline = load_pipeline()
    
# Sidebar (Konfigurasi)
with st.sidebar:
    st.header("Konfigurasi")
    
    rag_mode = st.selectbox(
        "Pilih Pendekatan RAG:",
        ("hybrid", "vector", "graph"),
        index=0,
        format_func=lambda x: x.upper() + "-RAG"
    )
    top_k = st.slider("Jumlah Konteks (Top-K):", min_value=1, max_value=10, value=3)
    
    st.divider()
    st.info(
        "**Keterangan Mode:**\n\n"
        "🔹 **Vector:** Pencarian semantik (FAISS).\n"
        "🔸 **Graph:** Penelusuran relasi entitas (REBEL).\n"
        "🔹🔸 **Hybrid:** Gabungan keduanya."
    )
    if st.button("Bersihkan Chat"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
 
# Tampilkan histori chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Jika ada konteks/sumber, tampilkan dalam expander
        if "contexts" in message:
            with st.expander("Lihat Bukti/Konteks Pendukung"):
                for ctx in message["contexts"]:
                    st.markdown(f"**[{ctx.get('title', 'Unknown')}]**")
                    st.text(ctx.get('text', '')[:300] + "...")
                    st.divider()

# Input Pengguna
if prompt := st.chat_input("Tanyakan sesuatu tentang dataset HotpotQA..."):
    # Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Proses jawaban
    with st.chat_message("assistant"):
        with st.spinner(f"Sedang berpikir (Mode: {rag_mode.upper()})"):
            try:
                # Panggil pipeline
                result = pipeline.answer_question(query=prompt, mode=rag_mode, top_k=top_k)
                
                answer_text = result['answer']
                latency = result['latency']
                contexts = result['contexts']
                
                # Tampilkan jawaban
                st.markdown(answer_text)
                st.caption(f"Latency: {latency:.2f} detik")
                
                # Tampilkan sumber di UI
                with st.expander("Lihat Bukti/Konteks Pendukung"):
                    for ctx in contexts:
                        score_info = f"(Score: {ctx.get('score', 0):.3f})" if 'score' in ctx else ""
                        st.markdown(f"**[{ctx.get('title', 'Unknown')}]** {score_info}")
                        st.markdown(f"<div class='source-box'>{ctx.get('text','')}</div>", unsafe_allow_html=True)
                
                # Simpan ke histori
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "contexts": contexts
                })
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

                
