import os
from typing import TypedDict, Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(BASE_DIR, ".env"))

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

MAX_LOOPS = 3  # total critique-failures allowed (generation OR retrieval)


# --- At top of self_rag.py ---
try:
    from rag.retriever import HybridRetriever
    from rag.exa_tool import ExaSearchTool
except ImportError:
    from src.rag.retriever import HybridRetriever
    from src.rag.exa_tool import ExaSearchTool


_SHARED_RETRIEVER = None
_SHARED_EXA = None

def register_retrieval_tools(retriever_instance=None, exa_instance=None):
    """Allows controller.py to pass in already-initialized tool instances, preventing SQLite/RocksDB lock collisions."""
    global _SHARED_RETRIEVER, _SHARED_EXA
    if retriever_instance is not None:
        _SHARED_RETRIEVER = retriever_instance
    if exa_instance is not None:
        _SHARED_EXA = exa_instance

def retrieve(query: str, k: int = 4) -> list[str]:
    """Dynamically re-retrieves missing information via the shared Qdrant instance, falling back to Exa."""
    chunks = []
    
    # 1. Search local Qdrant using the shared active connection
    if _SHARED_RETRIEVER is not None:
        try:
            results = _SHARED_RETRIEVER.search(query=query, limit=k)
            for r in results:
                text = r.get("text", "").strip()
                if text:
                    chunks.append(f"[{r.get('title', 'PIB Record')}]\n{text}")
        except Exception as e:
            print(f"[!] Qdrant re-retrieval error: {e}")

    # 2. If Qdrant returns no chunks, fall back to Exa
    if not chunks and _SHARED_EXA is not None:
        try:
            print(f"[*] Self-RAG loop fetching live web supplement via Exa: '{query}'...")
            web_hits = _SHARED_EXA.search(query=query, num_results=2)
            for h in web_hits:
                chunks.append(f"[Web: {h['title']} ({h['url']})]\n{h['text']}")
        except Exception as e:
            print(f"[!] Exa re-retrieval error: {e}")

    return chunks

# ---------------------------------------------------------------------------
# 1. Critique schema: now DIAGNOSES why the draft failed
# ---------------------------------------------------------------------------
class CritiqueGrade(BaseModel):
    is_grounded: bool = Field(
        description="True if every claim in the draft is supported by the context."
    )
    is_relevant: bool = Field(
        description="True if the draft addresses the user's core intent."
    )
    is_complete: bool = Field(
        description="True only if the specific data requested (numbers, rates, dates, names) is actually provided."
    )
    failure_type: Literal["none", "extraction_miss", "retrieval_gap", "hallucination"] = Field(
        description=(
            "none: draft is fine. "
            "extraction_miss: the requested data IS present in the context but the draft missed it. "
            "retrieval_gap: the requested data is NOT present anywhere in the context (draft may honestly say so). "
            "hallucination: draft contains claims not supported by the context."
        )
    )
    missing_info: str = Field(
        default="",
        description="Precisely what information is still missing (e.g. 'per-crop MSP figures in Rs/quintal for KMS 2025-26').",
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="If failure_type is retrieval_gap: 1-3 NEW, specific search queries likely to surface the missing data. Must differ from earlier queries.",
    )
    unsupported_claims: list[str] = Field(default_factory=list)
    feedback: str = Field(description="Concrete instruction for the next attempt.")


# ---------------------------------------------------------------------------
# 2. State
# ---------------------------------------------------------------------------
class SelfRAGState(TypedDict, total=False):
    query: str
    context: str
    seen_chunks: list[str]
    tried_queries: list[str]
    draft: str
    critique: Optional[CritiqueGrade]
    loops: int
    final_output: str
    status: str


# ---------------------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)
critique_llm = llm.with_structured_output(CritiqueGrade, method="json_schema")


def extract_text_content(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(item.text)
        return "".join(parts).strip()
    return str(content).strip()


# ---------------------------------------------------------------------------
# 4. Nodes
# ---------------------------------------------------------------------------
def generator_node(state: SelfRAGState) -> dict:
    critique = state.get("critique")
    feedback_prompt = ""
    if critique and critique.failure_type in ("extraction_miss", "hallucination", "retrieval_gap"):
        feedback_prompt = (
            "\n\n[REVIEWER FEEDBACK ON YOUR PREVIOUS DRAFT]\n"
            f"- Problem type: {critique.failure_type}\n"
            f"- Feedback: {critique.feedback}\n"
        )
        if critique.unsupported_claims:
            feedback_prompt += f"- Remove: {'; '.join(critique.unsupported_claims)}\n"

    system_msg = SystemMessage(
        content=(
            "You are an expert Indian Governance Intelligence Assistant.\n"
            "Answer ONLY from the provided context. Never invent dates, numbers, or schemes.\n"
            "If the context lacks the specific figures requested, say exactly which figures "
            "are missing, then share only what the context does support."
        )
    )
    human_msg = HumanMessage(
        content=f"USER QUERY:\n{state['query']}\n\nCONTEXT:\n{state['context']}{feedback_prompt}"
    )
    response = llm.invoke([system_msg, human_msg])
    return {"draft": extract_text_content(response.content)}


def critique_node(state: SelfRAGState) -> dict:
    prompt = (
        "CRITIQUE TASK. Diagnose the draft against the query and context.\n\n"
        f"USER QUERY:\n{state['query']}\n\n"
        f"SOURCE CONTEXT:\n{state['context']}\n\n"
        f"CANDIDATE DRAFT:\n{state['draft']}\n\n"
        f"QUERIES ALREADY TRIED: {state.get('tried_queries', [])}\n\n"
        "STEPS:\n"
        "1. Scan the CONTEXT itself for the specific data the user asked for.\n"
        "2. If it is in the context but the draft missed it -> failure_type=extraction_miss.\n"
        "3. If it is NOT in the context (regardless of what the draft says) -> failure_type=retrieval_gap, "
        "and propose 1-3 new, specific search_queries (e.g. use exact crop names, 'Rs per quintal', "
        "'Kharif Marketing Season 2025-26', table/annexure wording). Do not repeat tried queries.\n"
        "4. If the draft asserts things the context does not support -> failure_type=hallucination.\n"
        "5. Otherwise failure_type=none.\n"
        "An honest 'the figures are not in the context' is grounded, but still incomplete."
    )
    grade = critique_llm.invoke(prompt)

    passed = grade.is_grounded and grade.is_relevant and grade.is_complete
    return {
        "critique": grade,
        "loops": state.get("loops", 0) + (0 if passed else 1),
    }


def retrieve_node(state: SelfRAGState) -> dict:
    """Re-retrieve using the critique's rewritten queries and MERGE into context."""
    critique = state["critique"]
    seen = list(state.get("seen_chunks", []))
    tried = list(state.get("tried_queries", []))
    new_chunks = []

    for q in critique.search_queries[:3]:
        if q in tried:
            continue
        tried.append(q)
        for chunk in retrieve(q):
            if chunk not in seen:
                seen.append(chunk)
                new_chunks.append(chunk)

    context = state["context"]
    if new_chunks:
        context += "\n\n[ADDITIONAL RETRIEVED CONTEXT]\n" + "\n---\n".join(new_chunks)

    return {"context": context, "seen_chunks": seen, "tried_queries": tried}


def cite_and_respond_node(state: SelfRAGState) -> dict:
    return {"final_output": state["draft"], "status": "supported_and_relevant"}


def not_found_node(state: SelfRAGState) -> dict:
    """Retrieval kept failing: return an honest 'not found', not a padded answer."""
    missing = state["critique"].missing_info or "the specific figures requested"
    note = (
        f"\n\n> *Could not verify: {missing}. Searched with "
        f"{len(state.get('tried_queries', []))} additional queries without finding it in the knowledge base.*"
    )
    return {"final_output": state["draft"] + note, "status": "retrieval_gap"}


def best_effort_output_node(state: SelfRAGState) -> dict:
    disclaimer = "\n\n> *Notice: best-effort output; verification retries were exhausted.*"
    return {"final_output": state["draft"] + disclaimer, "status": "best_effort"}


# ---------------------------------------------------------------------------
# 5. Router: route by failure TYPE, not just a counter
# ---------------------------------------------------------------------------
def retry_gate(state: SelfRAGState) -> str:
    c = state["critique"]
    if c.is_grounded and c.is_relevant and c.is_complete:
        return "cite_and_respond"

    if state.get("loops", 0) >= MAX_LOOPS:
        return "not_found" if c.failure_type == "retrieval_gap" else "exhausted"

    if c.failure_type == "retrieval_gap" and c.search_queries:
        return "re_retrieve"

    return "regenerate"  # extraction_miss / hallucination


# ---------------------------------------------------------------------------
# 6. Graph
# ---------------------------------------------------------------------------
builder = StateGraph(SelfRAGState)
builder.add_node("generator", generator_node)
builder.add_node("critique", critique_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("cite_and_respond", cite_and_respond_node)
builder.add_node("not_found", not_found_node)
builder.add_node("best_effort_output", best_effort_output_node)

builder.add_edge(START, "generator")
builder.add_edge("generator", "critique")
builder.add_conditional_edges(
    "critique",
    retry_gate,
    {
        "cite_and_respond": "cite_and_respond",
        "regenerate": "generator",
        "re_retrieve": "retrieve",
        "not_found": "not_found",
        "exhausted": "best_effort_output",
    },
)
builder.add_edge("retrieve", "generator")
builder.add_edge("cite_and_respond", END)
builder.add_edge("not_found", END)
builder.add_edge("best_effort_output", END)

self_rag_app = builder.compile()