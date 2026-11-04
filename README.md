# DRAG: Distilling RAG for Small Language Models (Experimental Pipeline)

Dự án này tái lập (replicate) và mở rộng kiến trúc thực nghiệm của bài báo khoa học **"DRAG: Distilling RAG for Small Language Models"**. 

Mục tiêu cốt lõi của phương pháp là **chắt lọc (distill) tri thức từ một Teacher LLM mạnh mẽ trên đám mây** (như `GPT-4o` / `GPT-4o-mini`) thành hai dạng biểu diễn nén:
1. **Văn bản Bằng chứng (Textual Evidence)**
2. **Đồ thị Tri thức (Knowledge Graph Triples)**

Sau đó, ngữ cảnh chắt lọc này được cung cấp cho một **Student SLM nhỏ gọn** (như `Qwen2.5-3B-Instruct` / `Gemma-2-2B-it`) chạy **hoàn toàn cục bộ (offline)** trên thiết bị người dùng (tối ưu hóa cho Apple Silicon Mac M2/M3 với Metal `mps`), giúp tối ưu độ chính xác, giảm thiểu hiện tượng ảo giác (hallucination) và bảo toàn tính riêng tư dữ liệu người dùng.

---

## 📐 Kiến trúc Thuật toán & Pipeline Thực Nghiệm

```
  [User Question (PII)] 
           │
           ▼ (Chạy cục bộ trên thiết bị)
  [🔒 Privacy Filter: Local SLM + Regex]
           │
           ▼ (Câu hỏi đã ẩn danh - De-identified Query)
  ┌─────────────────────────────────────────────────────────────┐
  │ ☁️ CLOUD TEACHER LLM (GPT-4o-mini)                         │
  │   [Step 1] Sinh N = 10..20 Bằng chứng khách quan (Evidences) │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼ (Tải về cục bộ)
  ┌─────────────────────────────────────────────────────────────┐
  │ 💻 LOCAL EDGE PIPELINE (Mac M2 / MPS)                       │
  │   [Step 2] Ranker (SentenceTransformer Cosine Similarity)   │
  │            -> Lọc Top-K bằng chứng quan trọng nhất          │
  │   [Step 3] Graph Triples Extraction & Aggregation           │
  │            -> Trích xuất bộ ba (Entity A, Entity B, Rel)    │
  │   [Step 4] Student SLM Inference (Qwen2.5-3B-Instruct)      │
  │            -> Suy luận đáp án trắc nghiệm (Greedy Decoding) │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
                     [🎉 Final Prediction]
```

### Các bước thành phần chính:

1. **🔒 Privacy-Preserving De-identification (Bảo mật quyền riêng tư - Section 3.3 & Appendix G/I):**
   * Trước khi gửi bất kỳ câu hỏi nào lên Cloud Teacher LLM, hệ thống sử dụng Regex và **Student SLM chạy trực tiếp trên máy** để tự động nhận diện và khử bỏ các thông tin định danh cá nhân nhạy cảm (PII - Tên, Email, SĐT, Cơ quan/Học vị). Chỉ câu hỏi đã được làm sạch (de-identified query) mới được gửi lên Cloud.
2. **Step 1: Evidence Generation (Teacher LLM):**
   * Teacher LLM sinh ra $N = 10 \sim 20$ câu/văn bản bằng chứng khách quan xung quanh câu hỏi $q$ mà không làm lộ trực tiếp đáp án (theo System Prompt chuẩn phụ lục bài báo).
3. **Step 2: Evidence Ranking & Filtering (Semantic Search):**
   * Bộ mã hóa embedding cục bộ (`SentenceTransformer/all-mpnet-base-v2`) tính Cosine Similarity giữa embedding câu hỏi $e_q$ và từng evidence $e_i$: $\text{score}_i^{(\text{sim})} = \cos(e_i, e_q)$.
   * Lọc và giữ lại tập Top-$K$ ($K \in \{5, 10, 15, 20\}$) bằng chứng liên quan nhất.
4. **Step 3: Graph RAG Generation & Aggregation:**
   * Trích xuất các bộ ba quan hệ Thực thể - Quan hệ - Thực thể: $R_i = \{(a, b, r) \mid a, b \in V, r \in E\}$.
   * Gom nhóm đồ thị tri thức để loại bỏ thông tin dư thừa, tạo biểu diễn cấu trúc ngắn gọn.
5. **Step 4: Small-Scale LLM Evaluation (Student SLM):**
   * Kết hợp Top-$K$ Evidence và Knowledge Graph Triples thành Context đưa vào Student SLM.
   * Student SLM đưa ra đáp án trắc nghiệm ($\hat{y} \leftarrow M_{\text{small}}(q, \text{context})$) bằng chiến lược Greedy Decoding (`do_sample=False`).

---

## 📁 Cấu trúc Thư mục Dự án

```text
Distilling-RAG-for-Small-Language-Models/
├── data/                         # Thư mục lưu trữ dữ liệu thực nghiệm
│   ├── raw/                      # Dữ liệu gốc tải từ HuggingFace Hub
│   └── processed/                # Dữ liệu pilot đã trích xuất (.jsonl)
│       ├── arc_challenge_pilot.jsonl   # 100 câu hỏi trắc nghiệm khoa học ARC
│       └── medmcqa_pilot.jsonl          # 100 câu hỏi y khoa MedMCQA
├── outputs/                      # Lưu trữ kết quả trung gian và log thực nghiệm
│   ├── evidences/                # Dữ liệu cache bằng chứng từ Teacher LLM
│   │   └── teacher_cached_data.json
│   ├── graphs/                   # Dữ liệu đồ thị tri thức (triples)
│   └── eval_results/             # Kết quả đánh giá mô hình Student
│       ├── ablation_results.csv        # Kết quả Ablation Study dạng CSV
│       └── ablation_results.json       # Kết quả Ablation Study dạng JSON
├── src/                          # Mã nguồn chính của dự án (Modular Architecture)
│   ├── __init__.py
│   ├── config.py                 # Cấu hình biến môi trường, đường dẫn, hyper-parameters
│   ├── teacher/                  # Module giao tiếp Teacher LLM (OpenAI API)
│   │   ├── __init__.py
│   │   ├── generator.py          # [Step 1] Sinh N evidences từ Teacher
│   │   └── graph_extractor.py    # [Step 3] Trích xuất và nén Knowledge Graph Triples
│   ├── retriever/                # Module xếp hạng và lọc vector
│   │   ├── __init__.py
│   │   └── ranker.py             # [Step 2] Cosine Similarity với SentenceTransformer
│   ├── student/                  # Module quản lý mô hình Student SLM Local
│   │   ├── __init__.py
│   │   ├── local_runner.py       # [Step 4] Nạp weights & thực thi Inference trên GPU/MPS
│   │   └── evaluator.py          # Module đo lường các chỉ số Accuracy, F1
│   └── utils/                    # Các module tiện ích và bảo mật dữ liệu
│       ├── __init__.py           # Expose các hàm tiện ích
│       ├── privacy.py            # [Privacy Filter] Khử định danh PII (Regex + Local SLM)
│       ├── data_loader.py        # Hàm hỗ trợ đọc/ghi dữ liệu JSONL
│       └── metrics.py            # Tính toán các chỉ số độ chính xác
├── scripts/                      # Các kịch bản thực thi chính (CLI Scripts)
│   ├── prepare_data.py           # Tải & sinh tập dữ liệu pilot (100 mẫu/tập)
│   ├── run_pipeline.py           # Chạy Pipeline DRAG chế độ tương tác (Interactive CLI)
│   ├── run_drag_with_privacy_demo.py # Demo DRAG có tích hợp bộ lọc bảo mật PII
│   ├── run_ablation.py           # Chạy tự động toàn bộ thực nghiệm Ablation Study
│   └── run_baselines.py          # Chạy đánh giá so sánh với các baseline gốc
├── .env                          # Biến môi trường chứa OpenAI API Key
├── .gitignore                    # Loại trừ virtualenv, cache, keys khỏi git
├── requirements.txt              # Danh sách các thư viện Python
├── setup.py                      # Cấu hình package nội bộ (hỗ trợ pip install -e .)
└── README.md                     # Tài liệu hướng dẫn dự án
```

---

## ⚙️ Chi tiết Ý nghĩa Các Module và Hàm Chính

### 1. `src/config.py`
* **Ý nghĩa:** Quản lý tập trung cấu hình toàn dự án thông qua biến môi trường.
* **Các thông số chính:**
  * `OPENAI_API_KEY`: Khóa API gọi Teacher LLM.
  * `TEACHER_MODEL`: Tên mô hình Teacher (`gpt-4o-mini` hoặc `gpt-4o`).
  * `STUDENT_MODEL_ID`: ID mô hình Student local trên Hugging Face (`Qwen/Qwen2.5-3B-Instruct`).
  * `EMBEDDING_MODEL`: Mô hình vector embedding (`sentence-transformers/all-mpnet-base-v2`).
  * `N_EVIDENCES`: Số lượng evidence cần Teacher sinh ra ($N=10$ hoặc $20$).
  * `K_TOP`: Số lượng evidence và quan hệ giữ lại sau bước lọc ($K=5$).

### 2. `src/utils/privacy.py` *(Mới)*
* **`filter_pii_regex(text: str) -> str`**: Sử dụng biểu thức chính quy (Regex) tốc độ cao để lọc các thông tin PII định dạng chuẩn: Email, Số điện thoại, Tiền tố xưng hô & tên (`Dr.`, `Prof.`, `Mr.`, `Ms.`).
* **`filter_pii_slm(question: str) -> str`**: Đưa câu hỏi qua Student SLM chạy offline trên máy người dùng kèm system prompt khử định danh, viết lại câu hỏi hoàn toàn sạch thông tin cá nhân mà vẫn giữ nguyên bản chất khoa học trước khi gửi lên Cloud API.

### 3. `src/teacher/generator.py`
* **`generate_evidences(question: str, n: int) -> list[str]`**: 
  Gửi câu hỏi $q$ tới Teacher LLM với System Prompt chuẩn từ Phụ lục F của bài báo để tạo ra danh sách $N$ văn bản bằng chứng thô.

### 4. `src/retriever/ranker.py`
* **`rank_and_filter_evidences(question: str, evidences: list[str], top_k: int) -> list[str]`**:
  Chuyển đổi câu hỏi và danh sách evidences thành vector, tính toán điểm Cosine Similarity và trích xuất Top-$K$ bằng chứng liên quan nhất.

### 5. `src/teacher/graph_extractor.py`
* **`extract_graph_triples(evidences: list[str]) -> str`**:
  Yêu cầu Teacher LLM trích xuất các bộ ba thực thể - quan hệ dạng `(entity_a, entity_b) Relationship: "..."` từ Top-$K$ bằng chứng và gom nhóm thông tin.

### 6. `src/student/local_runner.py`
* **`get_student()`**: Áp dụng mô hình **Singleton**, chỉ nạp mô hình Student SLM (`Qwen2.5-3B-Instruct`) lên GPU Metal MPS/RAM đúng 1 lần duy nhất trong suốt vòng đời của phiên làm việc.
* **`evaluate_student(question: str, options: str, context: str) -> str`**:
  Ghép `context` (Evidence + Graph) với `question` và `options`, sau đó gọi `student_model.generate(..., do_sample=False)` để đưa ra đáp án trắc nghiệm dạng ký tự (`A`, `B`, `C`, hoặc `D`).

---

## 🚀 Hướng Dẫn Cài Đặt và Thực Thi

### 1. Khởi tạo Môi trường & Cài đặt Thư viện
Yêu cầu Python 3.9 - 3.11. Chạy các lệnh sau trong terminal:

```bash
# Di chuyển vào thư mục dự án
cd Distilling-RAG-for-Small-Language-Models

# Khởi tạo môi trường ảo (ví dụ tên là drag_env)
python3 -m venv drag_env
source drag_env/bin/activate

# Cài đặt dự án ở chế độ phát triển (Editable Package) và cài đặt dependencies
pip install -e .
pip install -r requirements.txt
```

### 2. Cấu hình Khóa OpenAI API
Tạo file `.env` tại thư mục gốc của dự án:

```bash
echo "OPENAI_API_KEY=your_actual_openai_api_key_here" > .env
```

### 3. Chuẩn bị Dữ liệu Pilot
Tải và tiền xử lý 100 mẫu dữ liệu từ `ai2_arc` (ARC-Challenge) và `medmcqa`:

```bash
python scripts/prepare_data.py
```
*Kết quả:* Các file `arc_challenge_pilot.jsonl` và `medmcqa_pilot.jsonl` được tạo tại `data/processed/`.

---

## 🧪 Các Kịch Bản Chạy Thực Nghiệm

### Kịch bản 1: Chạy Pipeline DRAG Tương tác (Interactive CLI)
Script này cho phép thử nghiệm trực quan từng câu hỏi với menu lựa chọn:

```bash
python scripts/run_pipeline.py
```

Menu điều khiển:
```text
=== MENU THỰC NGHIỆM ===
1. Chạy câu hỏi mẫu Demo
2. Chọn random câu hỏi từ dataset ARC-Challenge
3. Thoát chương trình
```
* Chọn `1`: Chạy câu hỏi mẫu demo có đầy đủ log từ Step 1 đến Step 4.
* Chọn `2`: Chọn ngẫu nhiên 1 câu hỏi từ tập `arc_challenge_pilot.jsonl`, hiển thị câu hỏi, các lựa chọn, ground truth và kết quả suy luận của Student SLM.

---

### Kịch bản 2: Demo Bảo Mật Quyền Riêng Tư (Privacy-Preserving DRAG) *(Mới)*
Minh họa luồng bảo mật theo Section 3.3 của bài báo: khi người dùng nhập câu hỏi có chứa thông tin cá nhân (PII), Student SLM trên máy sẽ tự động khử định danh trước khi gửi lên OpenAI API:

```bash
python scripts/run_drag_with_privacy_demo.py
```

*Quy trình thực hiện:*
1. **Câu hỏi gốc có PII:** Chứa tên bác sĩ, email giả lập, chức danh.
2. **Khử định danh (De-identification):** Local SLM lọc sạch PII nhưng giữ trọn vẹn ngữ nghĩa khoa học.
3. **Gọi Teacher LLM:** Chỉ gửi câu hỏi đã ẩn danh lên OpenAI.
4. **Suy luận & Trả về kết quả:** Đưa ra đáp án chính xác cho người dùng.

---

### Kịch bản 3: Chạy Toàn Bộ Ablation Study Tự Động *(Mới)*
Script thực hiện kiểm chứng thực nghiệm đa cấu hình và đa mức $K$:

```bash
python scripts/run_ablation.py
```

*Tính năng nổi bật:*
* **Cơ chế Caching Teacher LLM (`outputs/evidences/teacher_cached_data.json`):** Chỉ gọi API Teacher 1 lần duy nhất cho 100 câu hỏi. Các lần chạy đánh giá sau sẽ đọc từ cache, tiết kiệm tối đa chi phí API và thời gian chờ.
* **Đánh giá 4 cấu hình:**
  1. `original`: Student SLM thuần túy (không có context bổ sung).
  2. `evidence_only`: Chỉ cung cấp Top-$K$ Evidence dạng văn bản.
  3. `graph_only`: Chỉ cung cấp Knowledge Graph Triples.
  4. `combined`: Cung cấp đầy đủ cả Evidence và Knowledge Graph (Mô hình DRAG hoàn chỉnh).
* **Quét qua các mức:** $K \in \{5, 10, 15, 20\}$.
* **Tự động xuất báo cáo:** Lưu kết quả ra file [`outputs/eval_results/ablation_results.csv`](file:///Users/lahoangtu/Documents/TH%E1%BA%A0C%20S%C4%A8/GI%E1%BA%A2I%20THU%E1%BA%ACT%20N%C3%82NG%20CAO/SOURCE/RAGSmallLanguageModels/Distilling-RAG-for-Small-Language-Models/outputs/eval_results/ablation_results.csv) và [`outputs/eval_results/ablation_results.json`](file:///Users/lahoangtu/Documents/TH%E1%BA%A0C%20S%C4%A8/GI%E1%BA%A2I%20THU%E1%BA%ACT%20N%C3%82NG%20CAO/SOURCE/RAGSmallLanguageModels/Distilling-RAG-for-Small-Language-Models/outputs/eval_results/ablation_results.json).

---

## 📊 Bảng Kết Quả Thực Nghiệm Thực Tế (Ablation Study)

Kết quả thực nghiệm thu được khi chạy kiểm thử trên **100 mẫu câu hỏi trắc nghiệm khoa học ARC-Challenge** (`arc_challenge_pilot.jsonl`) với Student SLM `Qwen2.5-3B-Instruct`:

| $K$ | Cấu hình (Configuration) | Context Cung Cấp | Độ chính xác (Accuracy %) | Số câu đúng | Tổng mẫu |
|:---:|:---|:---|:---:|:---:|:---:|
| **0** | **`original`** *(Baseline)* | *Không context (SLM thuần)* | **85.0%** | 85 | 100 |
| 5 | `evidence_only` | Top-5 Evidences | 84.0% | 84 | 100 |
| 5 | `graph_only` | Knowledge Graph ($K=5$) | 83.0% | 83 | 100 |
| 5 | `combined` | Top-5 Evidences + Graph | 83.0% | 83 | 100 |
| **10** | **`evidence_only`** | Top-10 Evidences | **92.0%** | **92** | 100 |
| 10 | `graph_only` | Knowledge Graph ($K=10$) | 88.0% | 88 | 100 |
| 10 | `combined` | Top-10 Evidences + Graph | 88.0% | 88 | 100 |
| 15 | `evidence_only` | Top-15 Evidences | 90.0% | 90 | 100 |
| 15 | `graph_only` | Knowledge Graph ($K=15$) | 83.0% | 83 | 100 |
| 15 | `combined` | Top-15 Evidences + Graph | 89.0% | 89 | 100 |
| 20 | `evidence_only` | Top-20 Evidences | 91.0% | 91 | 100 |
| 20 | `graph_only` | Knowledge Graph ($K=20$) | 86.0% | 86 | 100 |
| **20** | **`combined`** *(Full DRAG)* | Top-20 Evidences + Graph | **92.0%** | **92** | 100 |

### 🔍 Nhận xét & Phân tích Thực nghiệm:
1. **Hiệu quả vượt bậc của phương pháp chắt lọc (Distillation):**
   * Mô hình Student SLM gốc khi không có ngữ cảnh đạt độ chính xác **85.0%**.
   * Khi được bổ sung tri thức chắt lọc từ Teacher LLM với $K=10$ (`evidence_only`) và $K=20$ (`combined`), độ chính xác tăng vọt lên **92.0%** (+**7.0%** so với baseline).
2. **Ảnh hưởng của kích thước $K$:**
   * Ở mức $K=5$, lượng thông tin bằng chứng còn quá ngắn, chưa đủ bao quát câu hỏi phức tạp.
   * Khi nâng lên $K=10$ hoặc $K=20$, ngữ cảnh đạt độ bão hòa thông tin tối ưu, giúp SLM suy luận chính xác hơn đáng kể.
3. **Vai trò kết hợp của Knowledge Graph:**
   * Cấu hình đầy đủ `combined` đạt đỉnh **92.0%** ở $K=20$, chứng minh việc bổ sung đồ thị quan hệ thực thể hỗ trợ SLM liên kết logic tốt hơn đối với các câu hỏi đa bước (multi-hop reasoning).

---

## 📝 Lưu ý Kỹ thuật & Tối ưu Hệ thống

* **Khởi tạo Singleton & Lazy Loading:** Các mô hình trọng lượng lớn (`Qwen2.5-3B-Instruct` và `all-mpnet-base-v2`) chỉ nạp 1 lần duy nhất vào bộ nhớ thông qua hàm `get_student()` và `get_embedder()`, tránh tình trạng tràn RAM (Out of Memory).
* **Tối ưu hóa GPU Metal (MPS trên macOS):** Hệ thống tự động nhận diện thiết bị `torch.device("mps")` trên vi xử lý Apple Silicon M-series, sử dụng `torch.float16` để tối ưu tốc độ sinh token.
* **Greedy Decoding:** Student SLM sử dụng thiết lập `do_sample=False` để triệt tiêu tính ngẫu nhiên, đảm bảo các thực nghiệm lặp lại cho kết quả hoàn toàn nhất quán theo tiêu chuẩn nghiên cứu khoa học.
* **Tương thích LibreSSL trên macOS:** Pinned `urllib3<2.0.0` trong môi trường để ngăn ngừa triệt để các cảnh báo `NotOpenSSLWarning`.
