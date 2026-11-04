# src/utils/privacy.py
import re

__all__ = ["filter_pii_regex", "filter_pii_slm"]

def filter_pii_regex(text: str) -> str:
    """Làm sạch PII cơ bản bằng Regular Expressions (Email, SĐT, IP, Chức danh)"""
    # 1. Xóa Email
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[REDACTED_EMAIL]', text)
    # 2. Xóa Số điện thoại
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[REDACTED_PHONE]', text)
    # 3. Xóa thông tin liên hệ như Dr. ..., Prof. ...
    text = re.sub(r'(Dr\.|Prof\.|Mr\.|Ms\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+', '[REDACTED_PERSON]', text)
    return text


def filter_pii_slm(question: str) -> str:
    """Dùng Student SLM chạy Local trên Mac M2 để viết lại câu hỏi đã lọc PII"""
    import torch
    from src.student.local_runner import get_student

    # Lọc sơ bộ bằng regex trước
    cleaned_q = filter_pii_regex(question)
    
    # Prompt chuẩn cho Student SLM làm sạch PII (Section 3.3 & Appendix G)
    prompt = (
        "You are a privacy-preserving assistant. Remove all personally identifiable information (PII) "
        "such as names, emails, affiliations, phone numbers, and addresses from the question below while "
        "preserving the core scientific/factual intent of the question. "
        "Output ONLY the cleaned question.\n\n"
        f"Original Question: {cleaned_q}\nCleaned Question:"
    )
    
    student_model, tokenizer = get_student()
    inputs = tokenizer(prompt, return_tensors="pt").to(student_model.device)
    
    with torch.no_grad():
        outputs = student_model.generate(**inputs, max_new_tokens=100, do_sample=False)
        
    result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return result if result else cleaned_q
