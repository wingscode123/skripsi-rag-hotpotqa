import torch
import sys

def check_gpu_status():
    print("\n=== 1. CEK PYTORCH GPU ===")
    try:
        print(f"PyTorch Version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ GPU Terdeteksi: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ VRAM: {vram:.2f} GB")
        else:
            print("❌ GPU TIDAK TERDETEKSI oleh PyTorch.")
    except Exception as e:
        print(f"❌ Error PyTorch: {e}")

    print("\n=== 2. CEK LLAMA-CPP (MISTRAL) ===")
    try:
        from llama_cpp import Llama
        print("Library llama_cpp ditemukan.")
        print("✅ Import berhasil. Perhatikan log di bawah:")
        print("   Jika muncul 'BLAS = 1', berarti GPU aktif.")
        print("   Jika muncul 'BLAS = 0', berarti hanya CPU.")
        
    except ImportError:
        print("❌ llama_cpp tidak terinstal dengan benar.")
    except Exception as e:
        print(f"❌ Error Llama: {e}")

    print("\n=== 3. CEK SPACY (NER) ===")
    try:
        import spacy
        import numpy
        print(f"✅ Spacy Version: {spacy.__version__}")
        print(f"✅ Numpy Version: {numpy.__version__} (Harus < 2.0.0)")
    except ImportError:
        print("❌ Spacy/Numpy bermasalah.")
    except Exception as e:
        print(f"❌ Error Spacy: {e}")

if __name__ == "__main__":
    check_gpu_status()