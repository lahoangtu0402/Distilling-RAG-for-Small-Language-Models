# src/teacher/graph_extractor.py
from openai import OpenAI
from src.config import OPENAI_API_KEY, TEACHER_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def extract_graph_triples(evidences: list[str]) -> str:
    print(f"  [Graph] Đang trích xuất quan hệ đồ thị từ {len(evidences)} evidences...")
    
    prompt = (
        "Extract entity-relation-entity triples (head, relation, tail) from the text below. "
        "Format output as: Entity Pair: (entity_a, entity_b) Relationship: \"...\"\n\n"
        + "\n".join(evidences)
    )
    print(f"  Prompt của graph extractor: {prompt} \n")

    response = client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    //print("\nGraph extractor's response: \n", response)

    graph_output = response.choices[0].message.content.strip()
    
    print("  [Graph] Đồ thị Tri thức (Triples) trích xuất được:")
    for line in graph_output.split("\n"):
        if line.strip():
            print(f"    ├─ {line.strip()}")
            
    return graph_output