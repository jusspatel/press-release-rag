import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from retriever import HybridRetriever
from crag import CRAGEngine, CRAGStatus

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test_crag_flow():
    print(f"[*] Initializing {MODEL_ID} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=DEVICE,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()

    print("[*] Connecting to HybridRetriever...")
    retriever = HybridRetriever()
    crag = CRAGEngine(tokenizer=tokenizer, model=model, device=DEVICE)

    # -----------------------------------------------------------------------
    # Test Case 1: Grounded Policy Query (Should be 'correct' or 'ambiguous')
    # -----------------------------------------------------------------------
    query_1 = "What is the strategic agenda and MoU between India and European Union regarding 6G technology?"
    print(f"\n=======================================================")
    print(f"[TEST 1] Query: {query_1}")
    print(f"=======================================================")

    chunks_1 = retriever.search(query_1, limit=3)
    print(f"[*] Retrieved {len(chunks_1)} chunks from Qdrant.")
    for i, c in enumerate(chunks_1):
        print(f"    - Chunk {i+1}: PRID {c.get('prid')} | Title: {c.get('title')[:60]}...")

    # Stage 1: Doc Evaluation
    print("\n--- Running Stage 1: Document Evaluation ---")
    doc_eval_1 = crag.evaluate_documents(query_1, chunks_1)
    print(f"Status              : {doc_eval_1.status.value}")
    print(f"Reasoning           : {doc_eval_1.reasoning}")
    print(f"Missing Facts       : {doc_eval_1.missing_facts}")
    print(f"Fallback Search Q   : {doc_eval_1.fallback_search_query}")

    # Stage 2: Decomposition & Strip Filtering
    print("\n--- Running Stage 2: Knowledge Decomposition & Strip Filtering ---")
    raw_strips_count = sum(len(crag.decompose(c.get("text", ""))) for c in chunks_1)
    refined_context = crag.filter_and_recompose(query_1, chunks_1)

    print(f"Total Raw Strips Extracted : {raw_strips_count}")
    print(f"\n[Recomposed Refined Context Output]:")
    print(refined_context)

    # -----------------------------------------------------------------------
    # Test Case 2: Out-of-Corpus / Misleading Query (Should trigger 'incorrect')
    # -----------------------------------------------------------------------
    query_2 = "What are the subsidies for electric two-wheelers under PM E-DRIVE scheme?"
    print(f"\n=======================================================")
    print(f"[TEST 2] Negative/Irrelevant Control: {query_2}")
    print(f"=======================================================")

    # Intentionally passing the 6G chunks to see if CRAG flags them as 'incorrect'
    doc_eval_2 = crag.evaluate_documents(query_2, chunks_1)
    print(f"Status              : {doc_eval_2.status.value}")
    print(f"Reasoning           : {doc_eval_2.reasoning}")
    print(f"Missing Facts       : {doc_eval_2.missing_facts}")
    print(f"Fallback Search Q   : {doc_eval_2.fallback_search_query}")

    retriever.close()


if __name__ == "__main__":
    test_crag_flow()