import os
import torch

# Lokasi absolut dari folder root project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path ke folder data
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

# Parameter Global
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Config] Perangkat yang digunakan: {DEVICE}")