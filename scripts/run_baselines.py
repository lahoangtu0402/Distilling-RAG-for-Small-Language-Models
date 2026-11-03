# scripts/run_baselines.py
import sys
from pathlib import Path

# Đảm bảo Python luôn nhận diện được thư mục gốc của project và package `src`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# TODO: Thêm logic đánh giá baseline models (Original SLM, etc.)

