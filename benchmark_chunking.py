import json
import time
import os
from pathlib import Path
from langchain_core.documents import Document
from src.parsers.pdf_parser import extract_documents_pdf
from src.rag.pipeline import build_index
from src.rag.retriever import build_hybrid_retriever

def load_benchmark_data():
    with open("tests/benchmark_data.json", "r", encoding="utf-8") as f:
        return json.load(f)

def run_benchmark():
    # Setup
    pdf_path = Path("chaybomoi.pdf")
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found.")
        return

    print("Extracting documents...")
    base_docs = extract_documents_pdf(pdf_path)
    
    benchmark_questions = load_benchmark_data()
    
    chunk_sizes = [500, 1000, 1500, 2000]
    overlaps = [50, 100, 200]
    
    results = []

    print(f"{'Size':<6} | {'Overlap':<8} | {'Avg Latency (ms)':<18} | {'Hit Rate (@3)':<12} | {'Chunks':<6}")
    print("-" * 60)

    for size in chunk_sizes:
        for overlap in overlaps:
            if overlap >= size:
                continue
                
            start_time = time.perf_counter()
            # 1. Build Index
            index = build_index(base_docs, chunk_size=size, overlap=overlap)
            
            # 2. Setup Hybrid Retriever (default k=3 for benchmark)
            # Dùng hybrid vì đây là mode mặc định giúp tăng độ chính xác
            retriever = build_hybrid_retriever(index.vectorstore, index.chunks, k=3)
            
            hits = 0
            latencies = []
            
            for item in benchmark_questions:
                q = item["question"]
                expected = item["expected_answer"]
                
                t0 = time.perf_counter()
                retrieved_docs = retriever.invoke(q)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)
                
                # Check if expected information is present in any retrieved chunk
                all_text = " ".join([d.page_content for d in retrieved_docs]).lower()
                
                # Cải thiện logic check: tìm các cụm từ quan trọng
                expected_keywords = [w for w in expected.lower().replace(",", "").replace(".", "").split() if len(w) > 3]
                if not expected_keywords:
                    expected_keywords = expected.lower().split()
                    
                matches = sum(1 for kw in expected_keywords if kw in all_text)
                
                if matches / len(expected_keywords) >= 0.5: # 50% match
                    hits += 1
            
            hit_rate = hits / len(benchmark_questions)
            avg_latency = sum(latencies) / len(latencies)
            
            results.append({
                "chunk_size": size,
                "overlap": overlap,
                "avg_latency_ms": round(avg_latency, 2),
                "hit_rate": round(hit_rate, 2),
                "chunk_count": len(index.chunks)
            })
            
            print(f"{size:<6} | {overlap:<8} | {avg_latency:<18.2f} | {hit_rate:<12.2f} | {len(index.chunks):<6}")

    # Save results to markdown for reporting
    with open("benchmark_results.md", "w", encoding="utf-8") as f:
        f.write("# Chunking Strategy Benchmark Results\n\n")
        f.write("| Chunk Size | Overlap | Avg Latency (ms) | Hit Rate (@3) | Chunk Count |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for r in results:
            f.write(f"| {r['chunk_size']} | {r['overlap']} | {r['avg_latency_ms']} | {r['hit_rate']} | {r['chunk_count']} |\n")
        f.write("\n\n**Note:** Hit Rate based on keyword matching (threshold >= 50%) in retrieved context @ k=3.")

    print("\nBenchmark completed. Results saved to benchmark_results.md")

if __name__ == "__main__":
    run_benchmark()
