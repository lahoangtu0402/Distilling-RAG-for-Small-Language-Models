# scripts/prepare_data.py
import sys
from pathlib import Path

# Đảm bảo Python luôn nhận diện được thư mục gốc của project và package `src`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from datasets import load_dataset
from src.config import DATA_DIR

def prepare_pilot_datasets():
    output_dir = DATA_DIR / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("⏳ Đang tải dữ liệu ARC-Challenge...")
    arc = load_dataset("ai2_arc", "ARC-Challenge", split="test")
    arc_samples = arc.shuffle(seed=42).select(range(100))
    
    print("⏳ Đang tải dữ liệu MedMCQA...")
    med = load_dataset("medmcqa", split="validation")
    med_samples = med.shuffle(seed=42).select(range(100))

    # Lưu dữ liệu vào data/processed/
    arc_samples.to_json(str(output_dir / "arc_challenge_pilot.jsonl"))
    med_samples.to_json(str(output_dir / "medmcqa_pilot.jsonl"))
    print("✅ Đã tạo xong dữ liệu pilot tại data/processed/!")

if __name__ == "__main__":
    prepare_pilot_datasets()