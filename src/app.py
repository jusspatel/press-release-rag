import streamlit as st
import time
from controller import orchestrate_query, RouteEnum
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

# ---------------------------------------------------------------------------
# Sidebar: System Status & Diagnostic Metrics
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/55/Emblem_of_India.svg", width=65)
    st.markdown("### **System Architecture**")
    st.markdown("""
    - **Front-Door Router:** Local Qwen 2.5 1.5B (CUDA)
    - **Primary Store:** Qdrant Hybrid RRF (`bge-large` + BM25)
    - **Dynamic Fallback:** Exa Neural Search
    - **Synthesis Engine:** Gemini 3.5 Flash (LangGraph Self-RAG)
    """)
    st.divider()

    st.markdown("### **Preset Governance Queries**")
    sample_queries = [
        "What is the strategic agenda and MoU between India and European Union regarding 6G technology?",
        "When was the original Digital India program first launched in 2015?",
        "Write a quick python function to implement quicksort.",
        "What are the approved MSP hike rates for Kharif crops?"
    ]
    
    selected_sample = st.selectbox("Load sample query:", ["Select..."] + sample_queries)
    st.caption("2026 Press Information Bureau (PIB) Official Intelligence System")

# ---------------------------------------------------------------------------
# Main Query Interface
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🏛️ Indian Governance Intelligence System</div>', unsafe_allow_html=True)
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
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            
            with m_col1:
                st.metric("Triage Route", result.get("route", "N/A").upper())
            with m_col2:
                source_formatted = result.get("source", "N/A").replace("_", " ").title()
                st.metric("Context Origin", source_formatted)
            with m_col3:
                # Formats 'retrieval_gap', 'supported_and_relevant', etc.
                raw_stat = result.get("evaluation_status", "N/A")
                graph_stat = raw_stat.replace("_", " ").title()
                st.metric("Critique Verification", graph_stat)
            with m_col4:
                loops_used = result.get("retries_used", 0)
                st.metric("Pipeline Latency", f"{elapsed_time:.2f}s", delta=f"{loops_used} Loops")
            st.divider()

            # Final Structured Output
            st.markdown("### **Intelligence Synthesis**")
            st.markdown(result.get("response", "No response returned."))

            # Expandable Raw Trace
            with st.expander("View Raw Pipeline Trace Payload"):
                st.json(result)

        except Exception as e:
            progress_bar.empty()
            st.error(f"Execution Error: {str(e)}")
            st.exception(e)

elif submit_clicked and not user_query.strip():
    st.warning("Please enter a query to run the intelligence pipeline.")