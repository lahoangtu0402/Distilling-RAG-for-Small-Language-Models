# src/retriever/ranker.py
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from src.config import EMBEDDING_MODEL

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"  [Ranker] Đang tải mô hình embedding ({EMBEDDING_MODEL}) lên {device}...")
embedder = SentenceTransformer(EMBEDDING_MODEL, device=str(device))

def rank_and_filter_evidences(question: str, evidences: list[str], top_k: int = 5) -> list[str]:
    if not evidences:
        print("  [Ranker] Không có evidence nào để xếp hạng!")
        return []
        
    print(f"  [Ranker] Đang tính toán Cosine Similarity cho {len(evidences)} evidences...")
    
    q_emb = embedder.encode(question, convert_to_tensor=True)
    d_embs = embedder.encode(evidences, convert_to_tensor=True)
    sim_scores = util.cos_sim(q_emb, d_embs)[0].cpu().numpy()

    # In điểm số tương đồng của từng evidence
    for idx, (score, ev) in enumerate(zip(sim_scores, evidences), 1):
        print(f"    ├─ [Score: {score:.4f}] {ev}")

    top_indices = np.argsort(sim_scores)[::-1][:top_k]
    filtered_evidences = [evidences[i] for i in top_indices]
    
    print(f"  [Ranker] Đã chọn Top {len(filtered_evidences)} evidences có điểm cao nhất:")
    for idx, ev in enumerate(filtered_evidences, 1):
        print(f"    ⭐ Top {idx}: {ev}")
        
    return filtered_evidences