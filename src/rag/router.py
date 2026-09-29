import json
import re
from enum import Enum
from typing import Optional
import torch
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# 1. System Prompt Definition (Binary Triage + Explicit Guardrails)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert Query Routing Controller and Metadata Extractor for an Indian governance intelligence system.
Analyze the user query and output a strictly formatted JSON object determining whether downstream document retrieval is required.

### ROUTING TARGETS:
1. `direct`:
   - USE THIS whenever retrieval from official government databases is UNNECESSARY.
   - Covers: Software engineering, programming/coding problems (e.g., Python, C++, algorithms like quicksort, debugging, scripts), mathematics, logic problems, general reasoning, creative drafting, translation, or conversational greetings/small talk.
   - RULE: If the user asks you to write code, implement an algorithm, solve a math problem, or answer general knowledge, you MUST select `direct`.

2. `vector_db`:
   - USE THIS ONLY for factual questions regarding the Government of India, public policies, Cabinet decisions, government schemes (e.g., PM-KISAN, PM Surya Ghar, PLI, Telecom/6G), budget allocations, gazette notifications, or national governance initiatives.
   - DO NOT route generic computer science, math, or coding queries here under any circumstances.

---

### EXAMPLES:
User: "Write a quick python function to implement quicksort."
Response:
{
  "route": "direct",
  "search_query": "",
  "ministry_filter": null,
  "confidence": 1.0,
  "reasoning": "Standard algorithmic coding task requiring direct generation without document retrieval."
}

User: "What are the latest MSP hike percentages approved by the cabinet for Kharif crops?"
Response:
{
  "route": "vector_db",
  "search_query": "latest MSP hike percentages approved cabinet Kharif crops",
  "ministry_filter": "Ministry of Agriculture and Farmers Welfare",
  "confidence": 0.98,
  "reasoning": "Inquires about official Indian cabinet decisions and agricultural policy."
}

User: "When was the original Digital India program first launched in 2015?"
Response:
{
  "route": "vector_db",
  "search_query": "Digital India program launch date 2015",
  "ministry_filter": "Ministry of Electronics and Information Technology",
  "confidence": 0.95,
  "reasoning": "Factual query regarding an official Government of India program and historical governance timeline."
}
"""

# ---------------------------------------------------------------------------
# 2. Schema Definitions
# ---------------------------------------------------------------------------
class RouteEnum(str, Enum):
    VECTOR_DB = "vector_db"
    DIRECT = "direct"


class RouterDecision(BaseModel):
    route: RouteEnum
    search_query: str = Field(
        description="Cleaned, keyword-dense search query stripped of conversational filler, or empty string if route is direct."
    )
    ministry_filter: Optional[str] = Field(
        default=None,
        description="Standardized name of the Ministry if explicitly mentioned or strongly implied, else null."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, 
        description="Model confidence score in the chosen route."
    )
    reasoning: str = Field(
        description="Single sentence explaining why this route was selected."
    )


# ---------------------------------------------------------------------------
# 3. Model Loading
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[*] Loading {MODEL_ID} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map=DEVICE,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)
model.eval()


def extract_json_block(text: str) -> str:
    """Extracts valid JSON block even if model wraps it in markdown code fences."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    return text.strip()


# ---------------------------------------------------------------------------
# 4. Routing Execution
# ---------------------------------------------------------------------------
def execute_routing(query: str, system_prompt: str = SYSTEM_PROMPT) -> RouterDecision:
    format_instruction = (
        "\n\nCRITICAL INSTRUCTIONS:\n"
        "1. Output ONLY a valid JSON instance. Do not output markdown text or explanations outside the JSON.\n"
        "2. Any programming, coding, algorithm, or math problem MUST have route='direct' and search_query=''.\n"
        "Required format:\n"
        "{\n"
        '  "route": "vector_db" | "direct",\n'
        '  "search_query": "cleaned keywords or empty string",\n'
        '  "ministry_filter": "Ministry Name or null",\n'
        '  "confidence": 0.95,\n'
        '  "reasoning": "One concise justification sentence."\n'
        "}"
    )
    
    messages = [
        {"role": "system", "content": system_prompt + format_instruction},
        {"role": "user", "content": query}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    json_str = extract_json_block(generated_text)
    
    try:
        return RouterDecision.model_validate_json(json_str)
    except Exception as e:
        print(f"\n[!] Failed to parse JSON. Raw output was:\n{generated_text}\n")
        raise e


# ---------------------------------------------------------------------------
# 5. Quick Verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_queries = [
        "What are the latest MSP hike percentages approved by the cabinet for Kharif crops?",
        "Write a quick python function to implement quicksort.",
        "When was the original Digital India program first launched in 2015?",
        "What is the strategic agenda and MoU between India and European Union regarding 6G technology?"
    ]

    for q in test_queries:
        res = execute_routing(q)
        print(f"\nQuery: {q}")
        print(f"  -> Route: {res.route.value} (conf: {res.confidence:.2f})")
        print(f"  -> Search Query: '{res.search_query}'")
        print(f"  -> Ministry Filter: {res.ministry_filter}")
        print(f"  -> Reason: {res.reasoning}")