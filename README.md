# Comparative Analysis of Vector-RAG, GraphRAG, and Hybrid-RAG for Multi-hop QA

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![FAISS](https://img.shields.io/badge/Search-FAISS-blueviolet)

This repository contains the source code and experimental data for the Undergraduate Thesis titled **"Analisis Komparatif Kinerja Arsitektur GraphRAG dan Vector-RAG untuk Multi-hop Question Answering pada Korpus Publik HotpotQA"** (Comparative Analysis of GraphRAG and Vector-RAG Performance for Multi-hop Question Answering on HotpotQA Corpus).

## Project Overview

Retrieval-Augmented Generation (RAG) is the standard for mitigating LLM hallucinations. While **Vector-RAG** is efficient, it often struggles with multi-hop reasoning. **GraphRAG** utilizes Knowledge Graphs to capture explicit relationships.

This project implements and compares three architectures:
1.  **Vector-RAG:** Dense retrieval using FAISS (IndexFlatIP).
2.  **GraphRAG:** Knowledge Graph construction using **REBEL** (Relation Extraction By End-to-end Language generation) and path-based traversal via NetworkX.
3.  **Hybrid-RAG:** Parallel fusion of vector and graph contexts.

**Key Dataset:** [HotpotQA](https://hotpotqa.github.io/) (Distractor Setting).

## System Architecture

The system is built end-to-end using Python. The pipeline consists of:

* **Generator:** Mistral-7B-Instruct-v0.2 (4-bit quantization).
* **Vector Store:** FAISS (Facebook AI Similarity Search).
* **Graph Construction:** REBEL-Large (Seq2Seq Model) + NetworkX.
* **Evaluation:** RAGAS (Faithfulness, Answer Relevancy) & Standard Metrics (F1, EM).

## Installation

### Prerequisites
* Python 3.10+
* CUDA-enabled GPU (Recommended: 24GB+ VRAM for Indexing, 12GB+ for Inference).

### Setup
1.  Clone the repository:
    ```bash
    git clone [https://github.com/username/repo-name.git](https://github.com/username/repo-name.git)
    cd repo-name
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Indexing (Data Construction)
To process the HotpotQA dataset, build the vector index, and construct the knowledge graph:

```bash
# Run for a subset (e.g., 20k documents)
python run_indexing.py --limit 20000 --suffix 20k_final --batch_size 64

# Run for full dataset (Background process recommended)
CUDA_VISIBLE_DEVICES=0 nohup python run_indexing.py --suffix full_dataset --batch_size 64 > indexing.log 2>&1 &