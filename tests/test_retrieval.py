from app.core.retriever_vector import VectorRetriever
from app.core.retriever_graph import GraphRetriever
from app.core.retriever_hybrid import HybridRetriever

def main():
    query = "What magazine was started first?"
    
    print(f"\n==== Test Vector RAG ====")
    vr = VectorRetriever()
    res_v = vr.retrieve(query, top_k=2)
    for r in res_v:
        print(f"- [{r['score']:.3f}] {r['text'][:100]}...")
        
        
    print(f"\n==== Test Graph RAG ====")
    gr = GraphRetriever()
    res_g = gr.retrieve(query)
    if res_g:
        for r in res_g:
            print(f"- [Graph] {r['text']}")
    else:
        print("- Tidak ada entitas query yang ditemukan di graf (wajar untuk sample kecil).")
    
    print(f"\n==== Test Hybrid RAG =====")
    hr = HybridRetriever()
    res_h = hr.retrieve(query, top_k=3)
    print(f"Total hasil gabungan: {len(res_h)}")

if __name__ == "__main__":
    main()