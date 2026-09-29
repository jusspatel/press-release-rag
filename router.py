import json
import re
from enum import Enum
from typing import Optional
import torch
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# 1. System Prompt Definition
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert Query Routing Controller and Metadata Extractor for an Indian governance intelligence system.
Your mission is to analyze user queries and output a strictly formatted JSON object determining the optimal downstream execution path.

### SYSTEM BOUNDARIES & KNOWLEDGE CONSTRAINTS:
1. Local Vector Database (`vector_db`):
   - Contains ONLY official Government of India Press Information Bureau (PIB) press releases published in the calendar year 2026.
   - Covers: Cabinet Committee on Economic Affairs (CCEA) decisions, ministerial notifications, central welfare schemes (e.g., PM-KISAN, PM Surya Ghar, PLI), budget announcements, gazette summaries, official MoUs, and bilateral governance summits from 2026.
   - Does NOT contain live streaming events, real-time market tickers, historical pre-2026 data, or non-governmental documentation.

2. Web Search (`web_search`):
   - Required for live or transient data: current commodity prices (gold, petrol, mandi rates), stock market indices (Sensex, Nifty), ongoing sports scores, live weather, or current breaking news.
   - Required for historical queries strictly prior to 2026 (e.g., "origins of the 1991 economic reforms", "2024 general election timeline").
   - Required for topics entirely outside government and public administration.

3. Direct Generation (`direct`):
   - Applied when NO retrieval is necessary.
   - Covers: General reasoning, mathematics, writing/editing assistance, translation, programming/coding problems, or conversational small talk/greetings.

---

### ROUTING & DISAMBIGUATION RULES:
- Temporal Anchor Rule:
  * If a query asks about an official Indian government scheme, policy, or cabinet approval without mentioning a year, default to `vector_db` under the assumption it targets current 2026 guidelines.
  * If a query explicitly specifies an earlier year (e.g., "2023 PM-KISAN changes"), route to `web_search`.
  * If a query explicitly specifies 2026, route to `vector_db`.

- Acronym & Scheme Priority:
  * Official Indian governance acronyms (e.g., MSP, CCEA, PLI, PM-Awas, MoA&FW, DGFT, UIDAI, NITI Aayog) strongly indicate `vector_db` unless asking for live market trading prices or historical origins.

- Search Query Optimization (`search_query`):
  * When routing to `vector_db` or `web_search`, extract the core semantic keywords. Strip filler phrases ("tell me about", "can you show me", "what is").
  * Retain specific numbers, scheme titles, ministries, and policy nouns.
  * For `direct`, leave `search_query` as an empty string.
"""

# ---------------------------------------------------------------------------
# 2. Schema Definitions
# ---------------------------------------------------------------------------
class RouteEnum(str, Enum):
    VECTOR_DB = "vector_db"
    WEB_SEARCH = "web_search"
    DIRECT = "direct"


class TemporalScope(str, Enum):
    YEAR_2026 = "2026"
    PRE_2026 = "pre-2026"
    REALTIME = "realtime"
    TIMELESS = "timeless"


class RouterDecision(BaseModel):
    route: RouteEnum
    search_query: str = Field(
        description="Cleaned, keyword-dense search query stripped of conversational filler, or empty string if route is direct."
    )
    ministry_filter: Optional[str] = Field(
        default=None,
        description="Standardized name of the Ministry if explicitly mentioned or strongly implied, else null."
    )
    temporal_scope: TemporalScope
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
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)
model.eval()


def extract_json_block(text: str) -> str:
    """Extracts valid JSON block even if model wraps it in markdown code fences."""
    # Match ```json { ... } ``` or ``` { ... } ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Match the outermost balanced curly braces
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    return text.strip()


# ---------------------------------------------------------------------------
# 4. Routing Execution
# ---------------------------------------------------------------------------
def execute_routing(query: str, system_prompt: str = SYSTEM_PROMPT) -> RouterDecision:
    # Explicit few-shot format template avoids generating schema metadata ($defs)
    format_instruction = (
        "\n\nCRITICAL: Respond ONLY with a valid JSON instance. Do not output markdown explanations or schemas.\n"
        "Required format:\n"
        "{\n"
        '  "route": "vector_db" | "web_search" | "direct",\n'
        '  "search_query": "cleaned keywords or empty string",\n'
        '  "ministry_filter": "Ministry Name or null",\n'
        '  "temporal_scope": "2026" | "pre-2026" | "realtime" | "timeless",\n'
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
            max_new_tokens=256,       # Ample headroom to prevent EOF errors
            temperature=0.01,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    json_str = extract_json_block(generated_text)
    
    try:
        return RouterDecision.model_validate_json(json_str)
    except Exception as e:
        print(f"\n[!] Failed to parse JSON. Raw output was:\n{generated_text}\n")
        raise e


# ---------------------------------------------------------------------------
# 5. Quick Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_queries = [
        "What are the latest MSP hike percentages approved by the cabinet for Kharif crops?",
        "What is the live gold rate in Delhi today?",
        "Write a quick python function to implement quicksort.",
        "When was the original Digital India program first launched in 2015?"
    ]

    for q in test_queries:
        res = execute_routing(q)
        print(f"\nQuery: {q}")
        print(f"  -> Route: {res.route.value} (conf: {res.confidence:.2f})")
        print(f"  -> Search Query: '{res.search_query}'")
        print(f"  -> Scope: {res.temporal_scope.value}")
        print(f"  -> Reason: {res.reasoning}")