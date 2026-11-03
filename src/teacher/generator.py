# src/teacher/generator.py
from openai import OpenAI
from src.config import OPENAI_API_KEY, TEACHER_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_evidences(question: str, n: int = 10) -> list[str]:
    print(f"  [Teacher] Đang gọi API ({TEACHER_MODEL}) để sinh {n} evidences...")
    
    system_prompt = (
        "You are an assistant in charge of generating factual evidence statements that aid in solving the provided question. "
        "Provide only the evidence statements with no additional remarks. Do not give the answer away directly."
    )
    print(f"  Prompt của teacher: {system_prompt}")
    
    user_prompt = f"Generate N={n} evidences that pertain to answering the following question: {question}"
    print(f"  Prompt của user: {user_prompt}")

    response = client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    raw_output = response.choices[0].message.content
    evidences = [line.strip("- ").strip() for line in raw_output.split("\n") if line.strip()]
    evidences = evidences[:n]
    
    print(f"  [Teacher] Đã sinh xong {len(evidences)} evidences.")
    for idx, ev in enumerate(evidences, 1):
        print(f"    ├─ Evidence {idx}: {ev}")
        
    return evidences