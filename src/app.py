import os
import time
from langsmith import Client
import streamlit as st
from controller import orchestrate_query, RouteEnum
langsmith_client = None
if os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY"):
    try:
        langsmith_client = Client()
    except Exception as e:
        langsmith_client = None

st.set_page_config(
    page_title="PIB Governance Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for governance dashboard
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #3B82F6;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def render_langsmith_thinking_trace(run_id_str: str):
    """Fetches run steps from LangSmith and renders each node's thinking process."""
    if not langsmith_client or not run_id_str:
        return

    st.markdown("### Pipeline Execution & Thinking Trace")
    
    with st.spinner("Fetching execution trace from LangSmith..."):
        try:
            # Brief delay to allow background telemetry ingestion
            time.sleep(1.2)
            
            project_name = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT") or "default"
            
            # Direct link to LangSmith Studio
            st.markdown(
                f"[Open Full Trace in LangSmith Studio](https://smith.langchain.com/projects/p/{project_name}/r/{run_id_str})"
            )

            # Query child runs without conflicting root/execution_order filters
            runs = list(
                langsmith_client.list_runs(
                    project_name=project_name,
                    filter=f'eq(trace_id, "{run_id_str}")'
                )
            )

            # If trace_id filter returns empty, fall back to querying by parent_run_id
            if not runs:
                runs = list(
                    langsmith_client.list_runs(
                        project_name=project_name,
                        parent_run_id=run_id_str
                    )
                )

            # Filter out the root orchestrator itself so only steps/nodes appear
            child_runs = [r for r in runs if str(r.id) != run_id_str]
            # Order steps chronologically
            child_runs.sort(key=lambda x: x.start_time if x.start_time else 0)

            if not child_runs:
                st.info("Trace recorded in LangSmith. Click the studio link above to inspect the execution tree.")
                return

            # Render individual thinking steps
            for i, r in enumerate(child_runs):
                step_name = r.name
                duration = f"({(r.end_time - r.start_time).total_seconds():.2f}s)" if (r.end_time and r.start_time) else ""
                
                with st.expander(f"Step {i+1}: **{step_name}** {duration}", expanded=(step_name in ["critique", "retrieve"])):
                    # Critique Node Inspection
                    if "critique" in step_name.lower():
                        outputs = r.outputs or {}
                        critique_data = outputs.get("critique", {})
                        if critique_data:
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.markdown(f"**Grounded:** `{critique_data.get('is_grounded')}`")
                            with c2:
                                st.markdown(f"**Relevant:** `{critique_data.get('is_relevant')}`")
                            with c3:
                                st.markdown(f"**Failure Type:** `{critique_data.get('failure_type')}`")
                            
                            st.markdown(f"**Feedback:** {critique_data.get('feedback')}")
                            
                            queries = critique_data.get("search_queries") or []
                            if queries:
                                st.markdown("**Generated Search Queries:**")
                                for q in queries:
                                    st.code(q, language="text")

                    # Retrieve Node Inspection
                    elif "retrieve" in step_name.lower():
                        outputs = r.outputs or {}
                        tried = outputs.get("tried_queries", [])
                        st.markdown(f"**Queries Tried:** `{tried}`")

                    # Generator Node Inspection
                    elif "generator" in step_name.lower():
                        outputs = r.outputs or {}
                        st.markdown("**Generated Candidate Draft:**")
                        st.caption(outputs.get("draft", ""))

                    # Raw Inputs/Outputs Inspect Popover
                    with st.popover("View Raw I/O"):
                        st.json({"inputs": r.inputs, "outputs": r.outputs})

        except Exception as e:
            st.caption(f"Could not load detailed LangSmith trace: {e}")
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/55/Emblem_of_India.svg", width=65)
    st.markdown("### **System Architecture**")
    st.markdown("""
    - **Front-Door Router:**
      - *Primary:* Gemini 3.5 Flash-Lite (Fast, zero-shot structured output)
      - *Offline Fallback:* Local Qwen 2.5 1.5B (CUDA/CPU)
      - *Pre-Guardrail:* 0ms deterministic code/math triage
    - **Primary Store:**
      - Qdrant Hybrid RRF (`BAAI/bge-large-en-v1.5` dense + `bm25` sparse)
    - **Evaluation & Compaction:**
      - CRAG Evaluator: Local Qwen 2.5 1.5B
      - Knowledge Compactor: Local Qwen 2.5 1.5B (Single-pass factual density)
    - **Dynamic Fallback:**
      - Exa Neural Web Search (External governance & historical archives)
    - **Synthesis & Reflection:**
      - Gemini 3.5 Flash (LangGraph Self-RAG multi-hop critique loop)
    - **Observability:**
      - LangSmith Real-Time Tracing & Telemetry
    """)
    st.divider()

    st.markdown("### **Preset Sample Queries**")
    sample_queries = [
        "Did ONGC sign an MoU in 2026 for Eklavya Model Residential Schools?",
        "What is the strategic agenda and MoU between India and European Union regarding 6G technology?",
        "When was the original Digital India program first launched in 2015?",
        "Write a quick python function to implement quicksort.",
        "What are the approved MSP hike rates for Kharif crops?"
    ]
    
    selected_sample = st.selectbox("Load sample query:", ["Select..."] + sample_queries)


# ---------------------------------------------------------------------------
# Main Query Interface
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">Press Releases 2026 Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hybrid Corrective RAG (CRAG) & Self-RAG Pipeline across PIB 2026 Records</div>', unsafe_allow_html=True)

# Initialize Session State
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

if selected_sample != "Select...":
    st.session_state.query_input = selected_sample

user_query = st.text_input(
    label="Ask a question on official policies, Cabinet decisions, or general tasks:",
    value=st.session_state.query_input,
    placeholder="e.g., What are the details of the Bharat 6G Alliance MoU with EU?"
)

col1, col2 = st.columns([1, 6])
with col1:
    submit_clicked = st.button("Submit Query", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("Clear", use_container_width=False)
    if clear_clicked:
        st.session_state.query_input = ""
        st.rerun()

# ---------------------------------------------------------------------------
# Pipeline Execution & Observability Cards
# ---------------------------------------------------------------------------
if submit_clicked and user_query.strip():
    progress_bar = st.progress(5, text="Initiating pipeline...")
    start_time = time.time()

    with st.spinner("Processing through Router, Retrieval, and Self-RAG graph..."):
        progress_bar.progress(25, text="Triage via local Qwen router...")
        try:
            # Execute the unified orchestrator
            result = orchestrate_query(user_query)
            progress_bar.progress(100, text="Synthesis complete.")
            elapsed_time = time.time() - start_time
            time.sleep(0.3)
            progress_bar.empty()

            # Diagnostic Metric Banners
            st.markdown("### **Execution Telemetry**")
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            
            with m_col1:
                st.metric("Triage Route", result.get("route", "N/A").upper())
            with m_col2:
                crag_stat = result.get("crag_status", "N/A").replace("_", " ").title()
                st.metric("CRAG Evaluation", crag_stat)
            with m_col3:
                source_formatted = result.get("source", "N/A").replace("_", " ").title()
                st.metric("Context Origin", source_formatted)
            with m_col4:
                raw_stat = result.get("evaluation_status", "N/A")
                graph_stat = raw_stat.replace("_", " ").title()
                st.metric("Critique Verification", graph_stat)
            with m_col5:
                loops_used = result.get("retries_used", 0)
                st.metric("Pipeline Latency", f"{elapsed_time:.2f}s", delta=f"{loops_used} Loops")
            st.divider()

            # Final Structured Output
            st.markdown("### **Intelligence Synthesis**")
            st.markdown(result.get("response", "No response returned."))

            # Verified Source Documents (rendered ONLY if CRAG evaluated as CORRECT)
            if result.get("crag_status") == "correct" and result.get("relevant_documents"):
                st.divider()
                st.markdown("### **Verified Source Documents (CRAG Passed)**")
                st.caption("The CRAG evaluator verified that these official Press Information Bureau records contain direct, sufficient factual evidence:")
                
                for idx, doc in enumerate(result["relevant_documents"], 1):
                    doc_title = doc.get("title") or f"PIB Record {idx}"
                    doc_ministry = doc.get("ministry") or "Government of India"
                    doc_prid = doc.get("prid") or "N/A"
                    doc_published = doc.get("published_at", "")[:10] if doc.get("published_at") else "2026"
                    doc_url = doc.get("url") or ""
                    
                    with st.expander(f"{idx}. {doc_title} ({doc_ministry})", expanded=(idx == 1)):
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown(f"**Ministry:** {doc_ministry}")
                        with col_b:
                            st.markdown(f"**Date:** {doc_published}")
                        with col_c:
                            st.markdown(f"**PRID:** `{doc_prid}`")
                        
                        if doc_url:
                            st.markdown(f"[Open Official Press Release ({doc_prid})]({doc_url})")
                        
                        snippet = doc.get("text", "").strip()
                        if snippet:
                            st.markdown("**Excerpt:**")
                            st.info(snippet)

            # LangSmith Execution Trace Render
            if result.get("run_id"):
                st.divider()
                render_langsmith_thinking_trace(result.get("run_id"))

            # Expandable Raw Payload Trace
            with st.expander("View Raw Pipeline Trace Payload"):
                st.json(result)

        except Exception as e:
            progress_bar.empty()
            st.error(f"Execution Error: {str(e)}")
            st.exception(e)

elif submit_clicked and not user_query.strip():
    st.warning("Please enter a query to run the intelligence pipeline.")