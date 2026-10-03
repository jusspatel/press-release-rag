import os
import re
from enum import Enum
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

import torch
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. System Prompt Definition (Governance, PSU, & MoU Guardrails)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert Query Routing Controller and Metadata Extractor for an Indian governance intelligence system.
Analyze the user query and output a strictly formatted JSON object determining whether downstream document retrieval from official Press Information Bureau (PIB) / Government records is required.

### ROUTING TARGETS:
1. `vector_db` (DEFAULT for any real-world, government, policy, PSU, or institutional inquiries):
   - ALWAYS choose `vector_db` if the query mentions or asks about:
     * Government policies, Cabinet decisions, notifications, subsidies, or national budgets.
     * Public Sector Undertakings (PSUs) such as ONGC, NTPC, BHEL, BSNL, IOCL, GAIL, Coal India, Indian Railways, ISRO, DRDO.
     * Memorandums of Understanding (MoUs), bilateral/multilateral agreements, partnerships, or tenders.
     * Government educational or welfare initiatives (e.g. Eklavya Model Residential Schools / EMRS, PM-KISAN, PLI, PM Surya Ghar).
     * Any query asking "did [organization] sign...", "what did [ministry] announce...", or referring to a government document/press release.
   - RULE: If in doubt, ALWAYS select `vector_db`.

2. `direct` (STRICTLY RESERVED FOR CODING, MATH, & CHAT):
   - ONLY choose `direct` if the user is asking to:
     * Write computer software, programming code, functions, or algorithms (e.g., Python, C++, quicksort, SQL query, debugging).
     * Solve a pure mathematical equation or logic riddle.
     * Engage in conversational greetings or small talk (e.g., "hello", "who are you").
   - NEVER choose `direct` for factual, historical, organizational, company, or policy questions.

### MINISTRY FILTER RULES:
- ONLY set `ministry_filter` if the user EXPLICITLY mentions an official Ministry name containing the word 'Ministry' or 'Department' (e.g., 'Ministry of Agriculture', 'Ministry of Tribal Affairs', 'Ministry of Finance').
- If the query mentions a company, PSU, scheme, or school (e.g., ONGC, Eklavya, PM-KISAN, BSNL), set `ministry_filter` to null! PSUs are NOT ministries, and schemes often involve cross-ministerial collaborations. Do NOT guess or invent a ministry.

---

### EXAMPLES:
User: "Write a quick python function to implement quicksort."
Response:
{
  "route": "direct",
  "search_query": "",
  "ministry_filter": null,
  "confidence": 1.0,
  "reasoning": "Standard algorithmic programming task requiring direct generation without document retrieval."
}

User: "did ongc sign an mou in 2026. referring to the document ongc signing an mou for eklavya resdiential school models"
Response:
{
  "route": "vector_db",
  "search_query": "ONGC MoU Eklavya Model Residential Schools",
  "ministry_filter": null,
  "confidence": 0.98,
  "reasoning": "Inquires about an official MoU signed by ONGC for Eklavya Residential Schools, requiring vector database retrieval."
}

User: "What are the latest MSP hike percentages approved by the cabinet for Kharif crops?"
Response:
{
  "route": "vector_db",
  "search_query": "latest MSP hike percentages approved cabinet Kharif crops",
  "ministry_filter": null,
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
        description="Standardized name of the Ministry ONLY if explicitly mentioned in query, else null."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence score in the chosen route."
    )
    reasoning: str = Field(
        description="Single sentence explaining why this route was selected."
    )


# ---------------------------------------------------------------------------
# 3. Model Engine Management
# ---------------------------------------------------------------------------
# Select routing engine: 'gemini' (preferred for high-precision zero-shot) or 'local' (Qwen 2.5 1.5B)
ROUTER_BACKEND = os.getenv("ROUTER_BACKEND", "gemini" if os.getenv("GOOGLE_API_KEY") else "local")
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_LOCAL_TOKENIZER = None
_LOCAL_MODEL = None
_GEMINI_ROUTER = None

def _get_local_model():
    """Lazily loads local Qwen model only when requested."""
    global _LOCAL_TOKENIZER, _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        print(f"[*] Loading local routing model {MODEL_ID} on {DEVICE}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
        _LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map=DEVICE,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        _LOCAL_MODEL.eval()
    return _LOCAL_TOKENIZER, _LOCAL_MODEL


def _get_gemini_router():
    """Lazily loads Gemini structured router."""
    global _GEMINI_ROUTER
    if _GEMINI_ROUTER is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_model_name = os.getenv("ROUTER_GEMINI_MODEL", "gemini-3.5-flash-lite")
        llm = ChatGoogleGenerativeAI(model=gemini_model_name)
        _GEMINI_ROUTER = llm.with_structured_output(RouterDecision)
    return _GEMINI_ROUTER


# ---------------------------------------------------------------------------
# 4. Deterministic Guardrails & Sanitization
# ---------------------------------------------------------------------------
GOV_INDICATORS = [
    "ongc", "mou", "pib", "cabinet", "scheme", "yojana", "eklavya", "emrs",
    "msp", "kharif", "rabi", "subsidy", "tribal", "bureau", "gazette",
    "nstfdc", "bsnl", "ntpc", "bhel", "railway", "isro", "drdo", "press release"
]

CODE_PATTERNS = [
    r"\b(quicksort|mergesort|binary search|dijkstra|fibonacci)\b",
    r"^(write|code|implement|create)\s+(a|an)?\s*(python|c\+\+|java|golang|rust|sql|function|script|algorithm)",
]


def sanitize_decision(query: str, decision: RouterDecision) -> RouterDecision:
    """Sanitizes metadata filters and enforces strict guardrails."""
    q_lower = query.lower()

    # 1. Clean ministry filter: must be an actual ministry, not a PSU/company or general topic
    if decision.ministry_filter:
        mf = decision.ministry_filter.strip()
        if not ("ministry" in mf.lower() or "department" in mf.lower()):
            # Filter was a company/PSU (e.g. 'ONGC Ltd') or topic -> strip to None so retrieval is not blocked
            decision.ministry_filter = None

    # 2. Guardrail for governance queries mistakenly routed to direct
    if decision.route == RouteEnum.DIRECT and any(k in q_lower for k in GOV_INDICATORS):
        decision.route = RouteEnum.VECTOR_DB
        if not decision.search_query:
            clean_q = re.sub(
                r"^(did|can|what|how|referring to|the document|tell me about)\s+",
                "",
                query,
                flags=re.IGNORECASE,
            )
            decision.search_query = " ".join(clean_q.split())
        decision.reasoning += " (Corrected to vector_db by governance keyword guardrail)"

    # 3. Guardrail for coding queries mistakenly routed to vector_db
    for pat in CODE_PATTERNS:
        if re.search(pat, q_lower):
            decision.route = RouteEnum.DIRECT
            decision.search_query = ""
            decision.ministry_filter = None
            break

    return decision


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
# 5. Routing Execution
# ---------------------------------------------------------------------------
def execute_routing(query: str, system_prompt: str = SYSTEM_PROMPT) -> RouterDecision:
    """Executes query triage with Gemini (primary) or local Qwen (fallback), with deterministic guardrails."""
    q_clean = query.strip()

    # Fast 0ms pre-guardrail for unambiguous coding algorithm requests
    for pat in CODE_PATTERNS:
        if re.search(pat, q_clean.lower()):
            return RouterDecision(
                route=RouteEnum.DIRECT,
                search_query="",
                ministry_filter=None,
                confidence=1.0,
                reasoning="Direct programming or algorithmic coding task (0ms pre-guardrail).",
            )

    # Lane 1: Gemini Cloud Router (Ultra-high accuracy, zero-shot structured output)
    if ROUTER_BACKEND.lower() == "gemini" and os.getenv("GOOGLE_API_KEY"):
        try:
            gemini_router = _get_gemini_router()
            raw_decision = gemini_router.invoke(q_clean)
            return sanitize_decision(q_clean, raw_decision)
        except Exception as e:
            print(f"[!] Gemini router encountered error ({e}). Falling back to local model...")

    # Lane 2: Local Qwen 2.5 1.5B Router
    tok, mod = _get_local_model()

    format_instruction = (
        "\n\nCRITICAL INSTRUCTIONS:\n"
        "1. Output ONLY a valid JSON instance. Do not output markdown text or explanations outside the JSON.\n"
        "2. Any programming, coding, algorithm, or math problem MUST have route='direct' and search_query=''.\n"
        "3. Any question regarding PSUs (e.g. ONGC), MoUs, schemes (e.g. Eklavya), or government records MUST have route='vector_db'.\n"
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
        {"role": "user", "content": q_clean},
    ]

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = mod.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    generated_text = tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    json_str = extract_json_block(generated_text)

    try:
        raw_decision = RouterDecision.model_validate_json(json_str)
        return sanitize_decision(q_clean, raw_decision)
    except Exception as e:
        print(f"\n[!] Failed to parse JSON from local router. Raw output was:\n{generated_text}\n")
        # Graceful fallback: if parsing fails, default to vector_db so knowledge is not lost
        return RouterDecision(
            route=RouteEnum.VECTOR_DB,
            search_query=q_clean,
            ministry_filter=None,
            confidence=0.7,
            reasoning=f"Fallback vector_db routing after parse error: {e}",
        )


# ---------------------------------------------------------------------------
# 6. Quick Verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_queries = [
        "did ongc sign an mou in 2026. referring to the document ongc signing an mou for eklavya resdiential school models",
        "did ongc sign an mou in 2026",
        "What are the latest MSP hike percentages approved by the cabinet for Kharif crops?",
        "Write a quick python function to implement quicksort.",
        "When was the original Digital India program first launched in 2015?",
        "What is the strategic agenda and MoU between India and European Union regarding 6G technology?",
    ]

    print(f"[*] Running router verification with backend: '{ROUTER_BACKEND}'\n")
    for q in test_queries:
        res = execute_routing(q)
        print(f"Query: {q}")
        print(f"  -> Route: {res.route.value} (conf: {res.confidence:.2f})")
        print(f"  -> Search Query: '{res.search_query}'")
        print(f"  -> Ministry Filter: {res.ministry_filter}")
        print(f"  -> Reason: {res.reasoning}\n")