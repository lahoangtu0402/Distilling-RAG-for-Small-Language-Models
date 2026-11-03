# src/student/local_runner.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import STUDENT_MODEL_ID

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
_STUDENT_MODEL = None
_STUDENT_TOKENIZER = None
def get_student():
    global _STUDENT_MODEL, _STUDENT_TOKENIZER
    if _STUDENT_MODEL is None or _STUDENT_TOKENIZER is None:
        print(f"  [Student SLM] Đang nạp mô hình {STUDENT_MODEL_ID} vào GPU Metal...")
        _STUDENT_TOKENIZER = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
        _STUDENT_MODEL = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    return _STUDENT_MODEL, _STUDENT_TOKENIZER
    
def evaluate_student(question: str, options: str, context: str = "") -> str:
    student_model, tokenizer = get_student()
    prompt = (
        f"Context information:\n{context}\n\n"
        f"Question: {question}\nOptions:\n{options}\n"
        f"Answer with ONLY the correct option letter (A, B, C, or D):"
    )

    print("\n  [Student SLM] --- Prompt đưa vào mô hình ---")
    print(prompt)
    print("  [Student SLM] ---------------------------------")
    print("  [Student SLM] Đang thực hiện suy luận (Inference)...")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = student_model.generate(
            **inputs, 
            max_new_tokens=5, 
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None
        )
    generated_tokens = outputs[0][input_length:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    clean_answer = ""
    for char in result:
        if char.upper() in ['A', 'B', 'C', 'D']:
            clean_answer = char.upper()
            break
            
    return clean_answer if clean_answer else result