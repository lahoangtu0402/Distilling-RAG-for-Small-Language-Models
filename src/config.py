# src/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Đường dẫn dự án
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Cấu hình API Keys & Models
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TEACHER_MODEL = "gpt-4o-mini" # Dùng GPT-4o-mini cho chạy pilot thử nghiệm
STUDENT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct" # Hoặc Gemma-2-2B-it
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Tham số thực nghiệm DRAG (Algorithm 1)
N_EVIDENCES = 10  # Số lượng evidence sinh ra từ Teacher (N)
K_TOP = 5         # Số lượng evidence/graph quan trọng lọc lại (K)