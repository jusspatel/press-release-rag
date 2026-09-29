import os
from typing import TypedDict, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(BASE_DIR, ".env"))

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------------
# 1. Structured Output Schema for Critique
# ---------------------------------------------------------------------------
class CritiqueGrade(BaseModel):
    is_grounded: bool = Field(
        description="True if all statements are factual and fully supported by the context without hallucination."
    )
    is_relevant: bool = Field(
        description="True if the response directly and adequately addresses the user query."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Specific claims or numbers not present in the context."
    )
    feedback: str = Field(
        description="Clear instructions on how to correct the draft based strictly on the context."
    )

# ---------------------------------------------------------------------------
# 2. Graph State
# ---------------------------------------------------------------------------
class SelfRAGState(TypedDict):
    query: str
    context: str
    draft: str
    critique: Optional[CritiqueGrade]
    retry_count: int
    final_output: str
    status: str

# ---------------------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.2,
)

# Use native json_schema instead of tool/function calling to eliminate AFC warning
critique_llm = llm.with_structured_output(CritiqueGrade, method="json_schema")

# ---------------------------------------------------------------------------
# Helper: Safe Content Normalization
# ---------------------------------------------------------------------------
def extract_text_content(content) -> str:
    """Safely extracts text regardless of whether content is str or a list of blocks."""
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
def self_rag_generator_node(state: SelfRAGState) -> dict:
    """Step: Self-RAG generator. Generates/regenerates draft response."""
    feedback_prompt = ""
    if state.get("critique") and state["critique"].feedback:
        feedback_prompt = (
            f"\n\n[CRITIQUE FEEDBACK FROM PREVIOUS ATTEMPT]:\n"
            f"- Issues: {state['critique'].feedback}\n"
            f"- Remove these unsupported claims: {', '.join(state['critique'].unsupported_claims)}\n"
            f"Fix these errors and rely strictly on the context."
        )

    system_msg = SystemMessage(
        content=(
            "You are an expert Indian Governance Intelligence Assistant.\n"
            "Answer the query based ONLY on the provided verified context.\n"
            "Cite facts directly from the context. Do not invent dates, numbers, or schemes."
        )
    )
    human_msg = HumanMessage(
        content=f"USER QUERY:\n{state['query']}\n\nCONTEXT:\n{state['context']}{feedback_prompt}"
    )

    response = llm.invoke([system_msg, human_msg])
    clean_draft = extract_text_content(response.content)
    return {"draft": clean_draft}


def critique_node(state: SelfRAGState) -> dict:
    """Step: Critique (grounded & relevant?). Evaluates draft against context."""
    prompt = (
        f"USER QUERY:\n{state['query']}\n\n"
        f"SOURCE CONTEXT:\n{state['context']}\n\n"
        f"CANDIDATE DRAFT:\n{state['draft']}\n\n"
        f"Evaluate whether the candidate draft is fully grounded in the source context and relevant to the query."
    )
    grade = critique_llm.invoke(prompt)
    
    # Increment retry counter only if validation failed
    increment = 1 if not (grade.is_grounded and grade.is_relevant) else 0
    return {
        "critique": grade,
        "retry_count": state.get("retry_count", 0) + increment,
    }


def cite_and_respond_node(state: SelfRAGState) -> dict:
    """Step: Cite & respond (supported & relevant)."""
    return {
        "final_output": state["draft"],
        "status": "supported_and_relevant",
    }


def best_effort_output_node(state: SelfRAGState) -> dict:
    """Step: Best-effort output (exhausted retries)."""
    disclaimer = "\n\n> *Notice: This response was generated under best-effort output as verification retries were exhausted.*"
    return {
        "final_output": state["draft"] + disclaimer,
        "status": "best_effort",
    }

# ---------------------------------------------------------------------------
# 5. Conditional Edge: Retry Gate
# ---------------------------------------------------------------------------
def retry_gate(state: SelfRAGState) -> str:
    """
    Decides routing based on critique evaluation and retry count:
    - Supported & relevant -> cite_and_respond
    - Not supported/relevant AND count < 2 -> retry (self_rag_generator)
    - Retries exhausted (count >= 2) -> exhausted (best_effort_output)
    """
    critique = state.get("critique")
    
    # Path: Supported & relevant
    if critique and critique.is_grounded and critique.is_relevant:
        return "cite_and_respond"
    
    # Path: Retry gate (count < 2?)[cite: 1]
    if state.get("retry_count", 0) < 2:
        return "retry"
    
    # Path: Exhausted retries[cite: 1]
    return "exhausted"

# ---------------------------------------------------------------------------
# 6. Graph Assembly
# ---------------------------------------------------------------------------
builder = StateGraph(SelfRAGState)

builder.add_node("self_rag_generator", self_rag_generator_node)
builder.add_node("critique", critique_node)
builder.add_node("cite_and_respond", cite_and_respond_node)
builder.add_node("best_effort_output", best_effort_output_node)

builder.add_edge(START, "self_rag_generator")
builder.add_edge("self_rag_generator", "critique")

builder.add_conditional_edges(
    "critique",
    retry_gate,
    {
        "cite_and_respond": "cite_and_respond",
        "retry": "self_rag_generator",
        "exhausted": "best_effort_output",
    },
)

builder.add_edge("cite_and_respond", END)
builder.add_edge("best_effort_output", END)

self_rag_app = builder.compile()