# DRAG: Distilling RAG for Small Language Models (Experimental Pipeline)

Dự án này tái lập (replicate) kiến trúc thực nghiệm của bài báo khoa học **"DRAG: Distilling RAG for Small Language Models"**. Mục tiêu của phương pháp là chắt lọc (distill) tri thức từ một **Teacher LLM** lớn (như `GPT-4o` / `GPT-4o-mini`) thành hai dạng: **Văn bản Bằng chứng (Textual Evidence)** và **Đồ thị Tri thức (Knowledge Graph Triples)**, sau đó cung cấp context nén này cho một **Student SLM** nhỏ (như `Qwen2.5-3B-Instruct` / `Gemma-2-2B-it`) chạy trực tiếp trên thiết bị local để tăng độ chính xác và giảm hiện tượng ảo giác (hallucination).

---

## 📐 Kiến trúc Thuật toán (Algorithm 1 Breakdown)

Pipeline thực nghiệm bao gồm 4 bước chính theo đúng mô hình DRAG:

1. **Step 1: Evidence Generation (Teacher LLM)**
   * Teacher LLM sinh ra $N = 10$ câu/văn bản bằng chứng khách quan xung quanh câu hỏi $q$ mà không cho biết trực tiếp đáp án.
2. **Step 2: Evidence Ranking & Filtering (Semantic Search)**
   * Trình mã hóa Vector (`SentenceTransformer`) tính điểm Cosine Similarity giữa embedding của câu hỏi $e_q$ và từng evidence $e_i$: $\text{score}_i^{(\text{sim})} = \cos(e_i, e_q)$.
   * Lọc và giữ lại tập Top-$K$ ($K = 5$) bằng chứng đạt điểm số cao nhất.
3. **Step 3: Graph RAG Generation & Aggregation**
   * Teacher LLM trích xuất các bộ ba quan hệ Thực thể - Quan hệ - Thực thể: $R_i = \{(a, b, r) \mid a, b \in V, r \in E\}$.
   * Áp dụng gom nhóm đồ thị đơn giản (Simple Graph Aggregation) để giảm sự dư thừa thông tin.
4. **Step 4: Small-Scale LLM Evaluation (Student SLM)**
   * Ghép Top-$K$ Evidence và Knowledge Graph Triples thành Context.
   * Student SLM đưa ra đáp án trắc nghiệm ($\hat{y} \leftarrow M_{\text{small}}(q, \text{context})$) bằng chiến lược Greedy Decoding.

---

## 📁 Cấu trúc Thư mục Dự án

```text
drag-research/
├── data/                         # Thư mục lưu trữ dữ liệu
│   ├── raw/                      # Dữ liệu thô tải về từ HuggingFace
│   └── processed/                # Dữ liệu pilot đã trích xuất (.jsonl)
│       ├── arc_challenge_pilot.jsonl
│       └── medmcqa_pilot.jsonl
├── outputs/                      # Lưu trữ kết quả trung gian và log đánh giá
│   ├── evidences/                # Log bằng chứng đã sinh
│   ├── graphs/                   # Log đồ thị tri thức
│   └── eval_results/             # Kết quả Accuracy & Metrics
├── src/                          # Mã nguồn chính của dự án (Modular Design)
│   ├── __init__.py
│   ├── config.py                 # File cấu hình biến môi trường, thông số K, N
│   ├── teacher/                  # Các module tương tác với Teacher LLM (OpenAI API)
│   │   ├── __init__.py
│   │   ├── generator.py          # [Step 1] Sinh N evidences
│   │   └── graph_extractor.py    # [Step 3] Trích xuất & gom nhóm Graph Triples
│   ├── retriever/                # Module xếp hạng & lọc vector
│   │   ├── __init__.py
│   │   └── ranker.py             # [Step 2] Cosine Similarity với SentenceTransformer
│   ├── student/                  # Module quản lý mô hình Student SLM Local
│   │   ├── __init__.py
│   │   ├── local_runner.py       # [Step 4] Load weights & thực hiện Inference trên MPS/GPU
│   │   └── evaluator.py          # [Tùy chọn] Đo lường Accuracy % toàn bộ tập test
│   └── utils/                    # Các hàm phụ trợ
│       ├── __init__.py
│       ├── data_loader.py        # Hàm hỗ trợ đọc/ghi JSONL
│       └── metrics.py            # Tính toán độ chính xác
├── scripts/                      # Đĩa điều khiển chính (CLI scripts)
│   ├── prepare_data.py           # Script tải & chuẩn bị tập Pilot (100 câu)
│   └── run_pipeline.py           # Script chạy Pipeline DRAG tương tác (Interactive Loop)
├── .env                          # File chứa OpenAI API Key
├── .gitignore
├── requirements.txt              # Danh sách các thư viện Python
└── README.md                     # Tài liệu hướng dẫn dự án
```

---

## ⚙️ Chi tiết Ý nghĩa Các Module và Hàm Chính

### 1. `src/config.py`
* **Ý nghĩa:** Quản lý tập trung các tham số cấu hình toàn dự án.
* **Các tham số chính:**
  * `OPENAI_API_KEY`: API Key dùng cho Teacher LLM.
  * `TEACHER_MODEL`: Tên mô hình Teacher (`gpt-4o-mini` hoặc `gpt-4o`).
  * `STUDENT_MODEL_ID`: ID mô hình Student local trên Hugging Face (`Qwen/Qwen2.5-3B-Instruct`).
  * `EMBEDDING_MODEL`: Mô hình vector embedding (`sentence-transformers/all-mpnet-base-v2`).
  * `N_EVIDENCES`: Số lượng evidence cần sinh ra từ Teacher ($N=10$).
  * `K_TOP`: Số lượng evidence và relationship giữ lại sau khi lọc ($K=5$).

### 2. `src/teacher/generator.py`
* **`generate_evidences(question: str, n: int) -> list[str]`**: 
  Gửi câu hỏi $q$ tới Teacher LLM với System Prompt chuẩn từ Phụ lục F của bài báo để tạo ra danh sách $N$ văn bản bằng chứng thô.

### 3. `src/retriever/ranker.py`
* **`get_embedder()`**: Khởi tạo Singleton cho `SentenceTransformer` chạy trên MPS (Apple Silicon GPU) hoặc CUDA.
* **`rank_and_filter_evidences(question: str, evidences: list[str], top_k: int) -> list[str]`**:
  Chuyển đổi `question` và `evidences` thành dạng vector, tính Cosine Similarity và trích xuất Top-$K$ bằng chứng liên quan nhất.

### 4. `src/teacher/graph_extractor.py`
* **`extract_graph_triples(evidences: list[str]) -> str`**:
  Nhận vào Top-$K$ bằng chứng, yêu cầu Teacher LLM trích xuất các bộ ba thực thể - quan hệ dạng `(entity_a, entity_b) Relationship: "..."` và thực hiện nén đồ thị.

### 5. `src/student/local_runner.py`
* **`get_student()`**: Khởi tạo Singleton nạp mô hình Student SLM (`Qwen2.5-3B-Instruct`) lên RAM/GPU 1 lần duy nhất để tránh việc load lại mô hình nhiều lần.
* **`evaluate_student(question: str, options: str, context: str) -> str`**:
  Ghép `context` (Evidence + Graph) với `question` và `options`, sau đó gọi `student_model.generate(..., do_sample=False)` để đưa ra đáp án dạng ký tự (A, B, C, hoặc D).

---

## 🚀 Hướng Dẫn Cài Đặt và Chạy Thực Nghiệm

### 1. Chuẩn bị Môi trường
Yêu cầu Python 3.10 trở lên. Mở Terminal và thực thi:

```bash
# Clone hoặc mở thư mục dự án
cd drag-research

# Khởi tạo virtualenv
python3 -m venv venv
source venv/activate

# Cài đặt các thư viện cần thiết
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Thiết lập Biến Môi trường API Key
Tạo file `.env` ở thư mục gốc của project và nhập OpenAI API Key của bạn:

```bash
echo "OPENAI_API_KEY=sk-proj-xxxxyour_actual_key_herexxxx" > .env
```

### 3. Tải & Chuẩn bị Tập Dữ liệu Pilot
Chạy script để trích xuất 100 câu hỏi mẫu từ `ARC-Challenge` và `MedMCQA`:

```bash
python scripts/prepare_data.py
```
*Kết quả:* Các file `jsonl` sẽ được lưu tại `data/processed/`.

### 4. Chạy Thực Nghiệm Pipeline Tương Tác (Interactive Mode)
Khởi chạy script điều khiển chính:

```bash
python scripts/run_pipeline.py
```

Sau khi các mô hình (`SentenceTransformer` và `Qwen2.5-3B-Instruct`) được nạp 1 lần duy nhất vào bộ nhớ (RAM/VRAM/MPS), màn hình sẽ xuất hiện Menu:

```text
=== MENU THỰC NGHIỆM DRAG ===
1. Chạy câu hỏi mẫu Demo
2. Lấy ngẫu nhiên 1 câu hỏi từ tập dữ liệu ARC-Challenge (jsonl)
3. Nhập câu hỏi và các lựa chọn thủ công từ bàn phím
4. Thoát chương trình
```

* Nhấn **`1`** để xem luồng chạy mẫu với đầy đủ log các bước từ Step 1 đến Step 4.
* Nhấn **`2`** để kiểm thử ngẫu nhiên các câu hỏi trong tập dữ liệu `ARC-Challenge`.

---

## 📝 Lưu ý Kỹ thuật
* **Xử lý bộ nhớ trên Apple Silicon (Mac M2/M3):** Code tự động nhận diện thiết bị `mps` (Metal Performance Shaders).
* **Tránh lỗi Sampling/Temperature:** Student SLM sử dụng thiết lập `do_sample=False` để kích hoạt chiến lược **Greedy Decoding**, giúp kết quả nhất quán đúng theo tiêu chuẩn đánh giá bài báo khoa học.