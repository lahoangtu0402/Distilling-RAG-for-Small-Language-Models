# scripts/run_ablation.py
import sys
from pathlib import Path

# Đảm bảo Python luôn nhận diện được thư mục gốc của project và package `src`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import os
import pandas as pd
from tqdm import tqdm
from src.config import DATA_DIR, OUTPUTS_DIR, N_EVIDENCES
from src.teacher.generator import generate_evidences
from src.retriever.ranker import rank_and_filter_evidences
from src.teacher.graph_extractor import extract_graph_triples
from src.student.local_runner import evaluate_student

def load_dataset(filename="arc_challenge_pilot.jsonl"):
    filepath = DATA_DIR / "processed" / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_context(config_type: str, filtered_evidences: list, graph_triples: str) -> str:
    if config_type == "original":
        return ""
    elif config_type == "evidence_only":
        return "Evidences:\n" + "\n".join(filtered_evidences)
    elif config_type == "graph_only":
        return "Knowledge Graph:\n" + graph_triples
    elif config_type == "combined":
        return "Evidences:\n" + "\n".join(filtered_evidences) + "\n\nKnowledge Graph:\n" + graph_triples
    return ""


def run_ablation_study():
    dataset = load_dataset("arc_challenge_pilot.jsonl")
    total_samples = len(dataset)
    
    k_list = [5, 10, 15, 20]
    configs = ["original", "evidence_only", "graph_only", "combined"]
    
    print(f"🚀 Bắt đầu Ablation Study trên {total_samples} mẫu dữ liệu...")

    # -------------------------------------------------------------
    # BƯỚC 1: CACHING & LƯU CACHED_DATA RA FILE JSON
    # -------------------------------------------------------------
    cache_file = OUTPUTS_DIR / "evidences" / "teacher_cached_data.json"
    os.makedirs(cache_file.parent, exist_ok=True)
    
    cached_data = []

    # Nếu đã có file cache từ trước -> Đọc lại luôn để tiết kiệm chi phí gọi API
    if os.path.exists(cache_file):
        print(f"📦 Phát hiện file cache sẵn có tại {cache_file}. Đang tải lại dữ liệu...")
        with open(cache_file, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
    else:
        print("⏳ Đang gọi Teacher LLM để tạo dữ liệu cached_data...")
        for item in tqdm(dataset, desc="Caching Teacher Outputs"):
            q = item["question"]
            labels = item["choices"]["label"]
            texts = item["choices"]["text"]
            opts = "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])
            ground_truth = item["answerKey"].strip()

            # Sinh tối đa N=20 evidences từ Teacher
            raw_evidences = generate_evidences(q, n=max(k_list))
            
            cached_data.append({
                "id": item.get("id", ""),
                "question": q,
                "options": opts,
                "ground_truth": ground_truth,
                "raw_evidences": raw_evidences
            })
            
        # Ghi cached_data ra file JSON
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cached_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã lưu file cached_data thành công tại: {cache_file}")

    # -------------------------------------------------------------
    # BƯỚC 2: CHẠY ĐÁNH GIÁ VÀ LƯU RESULTS
    # -------------------------------------------------------------
    results = []

    for k in k_list:
        print(f"\n=======================================================")
        print(f"🔍 ĐANG ĐÁNH GIÁ VỚI K = {k}")
        print(f"=======================================================")
        
        for config in configs:
            if config == "original" and k != k_list[0]:
                continue

            correct_count = 0
            
            for sample in tqdm(cached_data, desc=f"Config: {config} (K={k})"):
                q = sample["question"]
                opts = sample["options"]
                gt = sample["ground_truth"]
                
                if config == "original":
                    context = ""
                else:
                    top_k_evidences = rank_and_filter_evidences(q, sample["raw_evidences"], top_k=k)
                    graph_triples = extract_graph_triples(top_k_evidences)
                    context = build_context(config, top_k_evidences, graph_triples)

                pred = evaluate_student(q, opts, context=context)
                
                if pred.upper() == gt.upper():
                    correct_count += 1

            acc = (correct_count / total_samples) * 100
            print(f"  🎯 Result [{config}] (K={k}): Accuracy = {acc:.2f}% ({correct_count}/{total_samples})")
            
            results.append({
                "K": k if config != "original" else 0,
                "Configuration": config,
                "Accuracy (%)": round(acc, 2),
                "Correct": correct_count,
                "Total": total_samples
            })

    # -------------------------------------------------------------
    # BƯỚC 3: XUẤT LƯU RESULTS RA FILE CSV & JSON
    # -------------------------------------------------------------
    eval_dir = OUTPUTS_DIR / "eval_results"
    os.makedirs(eval_dir, exist_ok=True)
    
    # 1. Lưu dưới dạng CSV
    csv_path = eval_dir / "ablation_results.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    
    # 2. Lưu dưới dạng JSON
    json_results_path = eval_dir / "ablation_results.json"
    with open(json_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=======================================================")
    print("📊 BẢNG KẾT QUẢ ABLATION STUDY THU ĐƯỢC")
    print("=======================================================")
    print(df.to_markdown(index=False))
    print(f"\n✅ Đã lưu file kết quả CSV tại: {csv_path}")
    print(f"✅ Đã lưu file kết quả JSON tại: {json_results_path}")

if __name__ == "__main__":
    run_ablation_study()