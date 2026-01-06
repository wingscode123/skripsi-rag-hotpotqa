import os
import torch

# --- SETUP DIREKTORI DASAR ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path ke folder data mentah & model
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# --- SETUP DATA PROCESSED ---
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed", "batch_20k")
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

# --- DEFINISI NAMA FILE (SUFFIX 20K) ---
SUFFIX = "20k"

# Nama file spesifik
VECTOR_INDEX_NAME = f"hotpot_{SUFFIX}"                
METADATA_NAME     = f"hotpot_{SUFFIX}_meta.pkl"       
GRAPH_PKL_NAME    = f"knowledge_graph_{SUFFIX}.pkl"   
TRIPLETS_CSV_NAME = f"triplets_{SUFFIX}.csv"          
LINKING_CSV_NAME  = f"entity_linking_{SUFFIX}.csv"

# --- PARAMETER GLOBAL ---
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Setup Model LLM
LLM_MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
LLM_MODEL_PATH = os.path.join(MODEL_DIR, LLM_MODEL_FILE)

# Cek Device (GPU Laptop vs CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"==================================================")
print(f"[Config] Root Dir       : {BASE_DIR}")
print(f"[Config] Data Dir       : {DATA_PROCESSED_DIR}")
print(f"[Config] Target Suffix  : {SUFFIX}")
print(f"[Config] Perangkat      : {DEVICE}")
print(f"==================================================")