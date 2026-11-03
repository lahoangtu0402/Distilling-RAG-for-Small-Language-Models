import sys
from pathlib import Path
import json
import random
from src.config import DATA_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import N_EVIDENCES, K_TOP
from src.teacher.generator import generate_evidences
from src.retriever.ranker import rank_and_filter_evidences
from src.teacher.graph_extractor import extract_graph_triples
from src.student.local_runner import evaluate_student

def process_single_question(question: str, options: str):
    print("\n-------------------------------------------------------")
    print("📍 [STEP 1] Teacher LLM Sinh Evidences")
    print("-------------------------------------------------------")
    evidences = generate_evidences(question, n=N_EVIDENCES)

    print("\n-------------------------------------------------------")
    print("📍 [STEP 2] Ranker Lọc & Xếp Hạng Evidences")
    print("-------------------------------------------------------")
    filtered_evidences = rank_and_filter_evidences(question, evidences, top_k=K_TOP)

    print("\n-------------------------------------------------------")
    print("📍 [STEP 3] Teacher LLM Trích Xuất Đồ Thị (Knowledge Graph)")
    print("-------------------------------------------------------")
    graph_triples = extract_graph_triples(filtered_evidences)

    print("\n-------------------------------------------------------")
    print("📍 [STEP 4] Student SLM Suy Luận Đáp Án")
    print("-------------------------------------------------------")
    context = "Evidences:\n" + "\n".join(filtered_evidences) + "\n\nKnowledge Graph:\n" + graph_triples
    prediction = evaluate_student(question, options, context=context)

    print("\n=======================================================")
    print(f"🎉 KẾT QUẢ CUỐI CÙNG: {prediction}")
    print("=======================================================\n")

if __name__ == "__main__":
    print("\n🚀 Khởi tạo hệ thống DRAG Pipeline (Đang nạp các mô hình vào Memory)...")
    
    # Lúc này các model ở ranker.py và local_runner.py đã được nạp xong 1 lần duy nhất!
    
    # Vòng lặp liên tục để thử nghiệm nhiều câu hỏi
    while True:
        print("\n=== MENU THỰC NGHIỆM ===")
        print("1. Chạy câu hỏi mẫu Demo")
        print("2. Chọn random câu hỏi")
        print("3. Thoát chương trình")
        
        choice = input("Nhập lựa chọn của bạn (1-3): ").strip()
        
        if choice == "1":
            q = "Which term best describes the life cycle of an insect that reaches the adult stage without being a pupa?"
            opts = "A. Incomplete metamorphosis\nB. Complete metamorphosis\nC. Alternation of generations\nD. Spontaneous mutation"
            process_single_question(q, opts)
            
        elif choice == "2":
            jsonl_path = DATA_DIR / "processed" / "arc_challenge_pilot.jsonl"
            
            try:
                # Đọc danh sách câu hỏi từ file jsonl
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    samples = [json.loads(line) for line in f]
                
                # Chọn ngẫu nhiên 1 câu hỏi từ dataset
                item = random.choice(samples)
                
                # 1. Lấy câu hỏi q
                q = item["question"]
                
                # 2. Format danh sách lựa chọn (choices) thành dạng string opts
                choices = item["choices"]
                # choices có dạng: {'text': ['Option A', 'Option B', ...], 'label': ['A', 'B', ...]}
                formatted_opts = []
                for label, text in zip(choices["label"], choices["text"]):
                    formatted_opts.append(f"{label}. {text}")
                
                opts = "\n".join(formatted_opts)
                
                print(f"\n🎲 Đã chọn câu hỏi: {q}")
                print(f"\nCác lựa chọn: \n {opts}")
                print(f"-> Đáp án đúng gốc (Ground Truth): {item['answerKey']}")        
                
                # Truyền q và opts vào pipeline
                process_single_question(q, opts)
                
            except FileNotFoundError:
                print(f"❌ Không tìm thấy file dữ liệu tại {jsonl_path}. Hãy chạy `python scripts/prepare_data.py` trước!")
                
        elif choice == "3":
            print("👋 Đã thoát chương trình.")
            break
        else:
            print("❌ Lựa chọn không hợp lệ, vui lòng chọn lại.")