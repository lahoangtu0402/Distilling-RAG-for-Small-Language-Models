# scripts/run_drag_with_privacy_demo.py
import sys
from pathlib import Path

# Đảm bảo Python luôn nhận diện được thư mục gốc của project và package `src`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import random
from src.config import DATA_DIR, N_EVIDENCES, K_TOP
from src.utils.privacy import filter_pii_slm
from src.teacher.generator import generate_evidences
from src.retriever.ranker import rank_and_filter_evidences
from src.teacher.graph_extractor import extract_graph_triples
from src.student.local_runner import evaluate_student

def run_drag_with_privacy_demo():
    # Giả lập câu hỏi chứa thông tin nhạy cảm PII (Appendix I.1)[cite: 1]
    raw_question = (
        "Which term best describes the life cycle of an insect that reaches the adult stage without being a pupa? "
        "Dr. Amelia Thompson (athompson93@biochemjournal.fake) mentioned this in her biology lecture."
    )
    options = "A. Incomplete metamorphosis\nB. Complete metamorphosis\nC. Alternation of generations\nD. Spontaneous mutation"

    print("\n=======================================================")
    print("🚀 BẮT ĐẦU CHẠY DRAG PIPELINE (CÓ PRIVACY FILTER)")
    print("=======================================================")
    print(f"❓ Câu hỏi gốc chứa PII:\n  {raw_question}\n")

    # -------------------------------------------------------------
    # 🔒 BƯỚC BẢO MẬT: PRIVACY FILTER (Chạy hoàn toàn trên Local Mac M2)[cite: 1]
    # -------------------------------------------------------------
    print("🔒 [Privacy Step] Đang làm sạch PII trên Student SLM Local...")
    deidentified_question = filter_pii_slm(raw_question)
    print(f"✅ Câu hỏi đã làm sạch (De-identified Query):\n  {deidentified_question}\n")

    # -------------------------------------------------------------
    # 🌐 CHỈ GỬI CÂU HỎI ĐÃ BẢO MẬT LÊN CLOUD TEACHER LLM[cite: 1]
    # -------------------------------------------------------------
    print("-------------------------------------------------------")
    print("📍 [STEP 1] Teacher LLM Sinh Evidences từ De-identified Query")
    print("-------------------------------------------------------")
    evidences = generate_evidences(deidentified_question, n=N_EVIDENCES)

    print("\n-------------------------------------------------------")
    print("📍 [STEP 2] Ranker Lọc & Xếp Hạng Evidences")
    print("-------------------------------------------------------")
    filtered_evidences = rank_and_filter_evidences(deidentified_question, evidences, top_k=K_TOP)

    print("\n-------------------------------------------------------")
    print("📍 [STEP 3] Teacher LLM Trích Xuất Knowledge Graph")
    print("-------------------------------------------------------")
    graph_triples = extract_graph_triples(filtered_evidences)

    print("\n-------------------------------------------------------")
    print("📍 [STEP 4] Student SLM Suy Luận Đáp Án")
    print("-------------------------------------------------------")
    context = "Evidences:\n" + "\n".join(filtered_evidences) + "\n\nKnowledge Graph:\n" + graph_triples
    prediction = evaluate_student(deidentified_question, options, context=context)

    print("\n=======================================================")
    print(f"🎉 KẾT QUẢ DỰ ĐOÁN CUỐI CÙNG: {prediction}")
    print("=======================================================\n")

if __name__ == "__main__":
    run_drag_with_privacy_demo()