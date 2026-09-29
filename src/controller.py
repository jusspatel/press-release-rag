import os
import sys
from pathlib import Path

# Add project root and src directory to sys.path so modules resolve whether executed directly or as a package
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
for p in (str(BASE_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from rag.router import execute_routing, RouteEnum
    from rag.retriever import HybridRetriever
    from rag.crag import CRAGEngine, CRAGStatus
    from rag.exa_tool import ExaSearchTool
    from rag.self_rag import self_rag_app,register_retrieval_tools
except ImportError:
    from src.rag.router import execute_routing, RouteEnum
    from src.rag.retriever import HybridRetriever
    from src.rag.crag import CRAGEngine, CRAGStatus
    from src.rag.exa_tool import ExaSearchTool
    from src.rag.self_rag import self_rag_app,register_retrieval_tools

# ---------------------------------------------------------------------------
# 1. Hardware & Local Model Initialization
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[*] Initializing local routing/eval model ({MODEL_ID}) on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map=DEVICE,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)
model.eval()

retriever = HybridRetriever()
crag = CRAGEngine(tokenizer=tokenizer, model=model, device=DEVICE)
exa = ExaSearchTool()

# Register singletons to prevent lock contention
register_retrieval_tools(retriever_instance=retriever, exa_instance=exa)

# ---------------------------------------------------------------------------
# 2. Main Orchestration Function
# ---------------------------------------------------------------------------
def orchestrate_query(user_query: str) -> dict:
    print(f"\n=================================================================")
    print(f"[*] Processing Query: '{user_query}'")
    print(f"=================================================================")

    # -----------------------------------------------------------------------
    # Step 1: Front-Door Routing (Binary: direct vs vector_db)
    # -----------------------------------------------------------------------
    print("[1/3] Executing Query Route Triage...")
    decision = execute_routing(user_query)
    print(f"      -> Route      : {decision.route.value} (Confidence: {decision.confidence:.2f})")
    print(f"      -> Cleaned Q  : '{decision.search_query}'")
    print(f"      -> Ministry   : {decision.ministry_filter}")
    print(f"      -> Reasoning  : {decision.reasoning}")

    # -----------------------------------------------------------------------
    # Lane A: Direct Generation (Coding, Math, Reasoning, Small Talk)
    # -----------------------------------------------------------------------
    if decision.route == RouteEnum.DIRECT:
        print("\n[*] Routing directly to local generator (no retrieval required)...")
        prompt = (
            f"<|im_start|>system\nYou are a precise and helpful assistant. Provide a direct, correct response.<|im_end|>\n"
            f"<|im_start|>user\n{user_query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return {
            "query": user_query,
            "route": "direct",
            "source": "local_parametric",
            "evaluation_status": "direct_bypass",
            "retries_used": 0,
            "response": response_text.strip(),
        }

    # -----------------------------------------------------------------------
    # Lane B: Vector Retrieval + Dynamic CRAG Fallback
    # -----------------------------------------------------------------------
    search_kw = decision.search_query or user_query
    print(f"\n[2/3] Querying Qdrant Hybrid Index: '{search_kw}'...")
    chunks = retriever.search(
        query=search_kw,
        limit=4,
        ministry_filter=decision.ministry_filter,
    )

    context_str = ""
    source_label = ""

    # Case B.1: Zero Chunks returned by local vector DB -> Trigger Exa
    if not chunks:
        print("[!] Local vector DB returned 0 results. Triggering corrective Exa Web Search...")
        web_hits = exa.search(search_kw, num_results=3)
        context_str = "\n\n".join(
            f"[Source: {h['title']} ({h['url']})]\n{h['text']}" for h in web_hits
        )
        source_label = "exa_fallback_zero_db_hits"

    # Case B.2: Evaluate Chunks with CRAG
    else:
        print(f"      -> Evaluating {len(chunks)} Chunks with CRAG Engine...")
        doc_eval = crag.evaluate_documents(user_query, chunks)
        print(f"      -> CRAG Grade : {doc_eval.status.value.upper()}")
        print(f"      -> Rationale  : {doc_eval.reasoning}")

        if doc_eval.status == CRAGStatus.CORRECT:
            print("      -> Document confirmed relevant. Decomposing & filtering knowledge strips...")
            context_str = crag.filter_and_recompose(user_query, chunks)
            source_label = "qdrant_pib_verified_strips"

        elif doc_eval.status == CRAGStatus.AMBIGUOUS:
            fallback_kw = doc_eval.fallback_search_query or search_kw
            print(f"      -> Partial facts detected. Supplementing via Exa: '{fallback_kw}'...")
            local_strips = crag.filter_and_recompose(user_query, chunks)
            web_hits = exa.search(fallback_kw, num_results=2)
            web_text = "\n\n".join(
                f"[Web Source: {h['title']} ({h['url']})]\n{h['text']}" for h in web_hits
            )
            context_str = f"--- Official PIB Context ---\n{local_strips}\n\n--- External Web Context ---\n{web_text}"
            source_label = "hybrid_pib_plus_exa"

        else:  # CRAGStatus.INCORRECT
            fallback_kw = doc_eval.fallback_search_query or search_kw
            print(f"      -> Local chunks deemed incorrect. Discarding & triggering Exa: '{fallback_kw}'...")
            web_hits = exa.search(fallback_kw, num_results=3)
            context_str = "\n\n".join(
                f"[Source: {h['title']} ({h['url']})]\n{h['text']}" for h in web_hits
            )
            source_label = "exa_fallback_irrelevant_chunks"

    # -----------------------------------------------------------------------
    # Step 3: Self-RAG LangGraph Synthesis Loop (Gemini 3.5 Flash)

    print(f"\n[3/3] Passing context to Gemini 3.5 Flash Self-RAG Graph (Source: {source_label})...")
    
    # State matches SelfRAGState in self_rag.py
    initial_graph_state = {
        "query": user_query,
        "context": context_str,
        "seen_chunks": [context_str] if context_str else [],
        "tried_queries": [search_kw],
        "draft": "",
        "critique": None,
        "loops": 0,
        "final_output": "",
        "status": "",
    }

    graph_result = self_rag_app.invoke(initial_graph_state)

    return {
        "query": user_query,
        "route": decision.route.value,
        "source": source_label,
        "evaluation_status": graph_result.get("status"),
        "retries_used": graph_result.get("loops", 0),  # mapped from new 'loops' key
        "response": graph_result.get("final_output"),
    }


# ---------------------------------------------------------------------------
# 3. Execution Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_queries = [
        # Direct coding test
        "Write a quick python function to implement quicksort.",
        # Local PIB test
        "What is the strategic agenda and MoU between India and European Union regarding 6G technology?",
        # Historical / external query (tests dynamic Exa triggering via CRAG)
        "When was the original Digital India program first launched in 2015?",
    ]

    try:
        for q in test_queries:
            res = orchestrate_query(q)
            print("\n------------------- FINAL RESULT -------------------")
            print(f"Route       : {res['route']}")
            print(f"Source      : {res['source']}")
            print(f"Graph Status: {res['evaluation_status']} (Retries: {res['retries_used']})")
            print(f"Response:\n{res['response']}")
            print("----------------------------------------------------\n")
    finally:
        retriever.close()