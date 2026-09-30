"""
ResearchLens - Grounded RAG Research Paper Assistant
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from services.chat_service import answer_question
from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.chunker import create_chunks
from ingestion.embedder import create_embeddings


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ResearchLens",
    page_icon="🔬",
    layout="wide",
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MAX_SOURCES = 5
MIN_RANKING = 0.3

UPLOAD_DIR = Path("data") / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SESSION STATE
# ============================================================

if "paper_info" not in st.session_state:
    st.session_state.paper_info = None

if "processing_done" not in st.session_state:
    st.session_state.processing_done = False


# ============================================================
# HEADER
# ============================================================

st.title("🔬 ResearchLens")

st.subheader(
    "Grounded Research Paper Question Answering"
)

st.write(
    "Upload a research paper and ask questions using "
    "Retrieval-Augmented Generation (RAG)."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write(f"**LLM Model:** `{MODEL_NAME}`")

    st.write(
        f"**Embedding Model:** `{EMBEDDING_MODEL}`"
    )

    st.write(
        f"**Maximum sources:** `{MAX_SOURCES}`"
    )

    st.write(
        f"**Minimum ranking:** `{MIN_RANKING}`"
    )

    st.divider()

    st.info(
        "ResearchLens retrieves relevant sections from "
        "the uploaded research paper and generates "
        "grounded answers using an LLM."
    )


# ============================================================
# RAG PIPELINE
# ============================================================

st.markdown("### 🔄 RAG Pipeline")

pipeline = [
    "📄 PDF Upload",
    "📝 Text Extraction",
    "✂️ Chunking",
    "🔢 Embeddings",
    "🗄️ Qdrant",
    "🎯 Semantic Ranking",
    "🤖 LLM Answer",
]

cols = st.columns(len(pipeline))

for col, step in zip(cols, pipeline):

    with col:

        st.info(step)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.divider()

st.header("📄 Upload Research Paper")

uploaded_file = st.file_uploader(
    "Choose a research paper",
    type=["pdf"],
    help="Upload a PDF research paper.",
)


# ============================================================
# PROCESS UPLOADED PDF
# ============================================================

if uploaded_file is not None:

    file_path = UPLOAD_DIR / uploaded_file.name

    try:

        # ----------------------------------------------------
        # SAVE PDF
        # ----------------------------------------------------

        with open(file_path, "wb") as f:

            f.write(
                uploaded_file.getbuffer()
            )

        file_size_mb = (
            file_path.stat().st_size
            / (1024 * 1024)
        )

        st.success(
            f"Research paper uploaded: "
            f"**{uploaded_file.name}**"
        )

        # ----------------------------------------------------
        # PROCESS ONLY WHEN NEW FILE IS UPLOADED
        # ----------------------------------------------------

        current_file = st.session_state.paper_info

        if (
            current_file is None
            or current_file["file_name"]
            != uploaded_file.name
        ):

            with st.status(
                "Processing research paper...",
                expanded=True
            ) as status:

                # ============================================
                # STEP 1 - TEXT EXTRACTION
                # ============================================

                st.write(
                    "📝 Extracting text from PDF..."
                )

                pages = extract_text_from_pdf(
                    str(file_path)
                )

                page_count = len(pages)

                total_characters = sum(
                    len(page.get("text", ""))
                    for page in pages
                )

                st.write(
                    f"✓ Extracted **{page_count} pages**"
                )

                # ============================================
                # STEP 2 - CHUNKING
                # ============================================

                st.write(
                    "✂️ Creating text chunks..."
                )

                chunks = create_chunks(
                    pages,
                    chunk_size=1000,
                    chunk_overlap=200,
                )

                chunk_count = len(chunks)

                st.write(
                    f"✓ Created **{chunk_count} chunks**"
                )

                # ============================================
                # STEP 3 - EMBEDDINGS
                # ============================================

                st.write(
                    "🔢 Generating embeddings..."
                )

                if chunks:

                    chunk_texts = [
                        chunk["text"]
                        for chunk in chunks
                    ]

                    embeddings = create_embeddings(
                        chunk_texts
                    )

                    embedding_count = len(
                        embeddings
                    )

                    embedding_dimension = (
                        embeddings.shape[1]
                        if len(embeddings.shape) > 1
                        else 0
                    )

                else:

                    embedding_count = 0
                    embedding_dimension = 0

                st.write(
                    f"✓ Generated **{embedding_count} embeddings**"
                )

                st.write(
                    f"✓ Vector dimension: "
                    f"**{embedding_dimension}**"
                )

                # ============================================
                # COMPLETE
                # ============================================

                status.update(
                    label="Research paper processing completed!",
                    state="complete",
                )

            # ------------------------------------------------
            # STORE INFORMATION
            # ------------------------------------------------

            st.session_state.paper_info = {

                "file_name": uploaded_file.name,

                "file_size_mb": file_size_mb,

                "page_count": page_count,

                "total_characters": total_characters,

                "chunk_count": chunk_count,

                "embedding_count": embedding_count,

                "embedding_dimension":
                    embedding_dimension,

                "embedding_model":
                    EMBEDDING_MODEL,
            }

            st.session_state.processing_done = True

        else:

            st.info(
                "This paper has already been processed "
                "in this session."
            )

    except Exception as e:

        st.error(
            f"❌ Error while processing PDF: {e}"
        )


# ============================================================
# PAPER INFORMATION
# ============================================================

if st.session_state.paper_info is not None:

    info = st.session_state.paper_info

    st.divider()

    st.header("📊 Research Paper Processing")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📄 Pages",
            info["page_count"],
        )

    with col2:

        st.metric(
            "✂️ Chunks",
            info["chunk_count"],
        )

    with col3:

        st.metric(
            "🔢 Embeddings",
            info["embedding_count"],
        )

    with col4:

        st.metric(
            "📐 Vector Dimension",
            info["embedding_dimension"],
        )

    st.caption(
        f"File: {info['file_name']} | "
        f"Size: {info['file_size_mb']:.2f} MB | "
        f"Extracted characters: "
        f"{info['total_characters']:,}"
    )


# ============================================================
# PIPELINE DETAILS
# ============================================================

if st.session_state.paper_info is not None:

    info = st.session_state.paper_info

    st.divider()

    st.subheader("🔍 Processing Details")

    with st.expander(
        "📄 1. PDF Upload",
        expanded=False,
    ):

        st.write(
            f"**File:** {info['file_name']}"
        )

        st.write(
            f"**Size:** "
            f"{info['file_size_mb']:.2f} MB"
        )

    with st.expander(
        "📝 2. Text Extraction",
        expanded=False,
    ):

        st.write(
            f"**Pages processed:** "
            f"{info['page_count']}"
        )

        st.write(
            f"**Characters extracted:** "
            f"{info['total_characters']:,}"
        )

    with st.expander(
        "✂️ 3. Chunking",
        expanded=False,
    ):

        st.write(
            f"**Chunks created:** "
            f"{info['chunk_count']}"
        )

        st.write(
            "**Chunk size:** 1000 characters"
        )

        st.write(
            "**Chunk overlap:** 200 characters"
        )

    with st.expander(
        "🔢 4. Embeddings",
        expanded=False,
    ):

        st.write(
            f"**Embedding model:** "
            f"{info['embedding_model']}"
        )

        st.write(
            f"**Embeddings generated:** "
            f"{info['embedding_count']}"
        )

        st.write(
            f"**Vector dimension:** "
            f"{info['embedding_dimension']}"
        )

    with st.expander(
        "🗄️ 5. Qdrant Vector Database",
        expanded=False,
    ):

        st.write(
            "The existing Qdrant vector-store and "
            "retrieval pipeline is used when a question "
            "is submitted."
        )

        st.info(
            "Qdrant retrieval details are shown below "
            "after asking a question."
        )


# ============================================================
# UPLOADED PAPERS
# ============================================================

uploaded_files = list(
    UPLOAD_DIR.glob("*.pdf")
)

if uploaded_files:

    st.divider()

    st.subheader("📚 Uploaded Papers")

    for pdf in uploaded_files:

        st.write(
            f"📄 **{pdf.name}**"
        )


# ============================================================
# QUESTION SECTION
# ============================================================

st.divider()

st.header("💬 Ask a Question")

question = st.text_area(
    "Enter your question",
    placeholder=(
        "Example: What are the limitations "
        "of the proposed system?"
    ),
    height=120,
)


# ============================================================
# ASK BUTTON
# ============================================================

ask_button = st.button(
    "🔍 Ask ResearchLens",
    type="primary",
    use_container_width=True,
)


# ============================================================
# ANSWER
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif not uploaded_files:

        st.warning(
            "Please upload a research paper first."
        )

    else:

        try:

            with st.spinner(
                "🔄 Searching Qdrant and generating answer..."
            ):

                result = answer_question(
                    question.strip()
                )

            # ------------------------------------------------
            # ANSWER
            # ------------------------------------------------

            st.markdown("## 💡 Answer")

            answer = result.get(
                "answer",
                "No answer generated.",
            )

            st.markdown(answer)

            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            sources = result.get(
                "sources",
                [],
            )

            st.markdown(
                "## 📚 Retrieved Sources"
            )

            if sources:

                # ============================================
                # RETRIEVAL SUMMARY
                # ============================================

                rankings = []

                for source in sources:

                    score = source.get(
                        "ranking_score",
                        0.0,
                    )

                    try:

                        rankings.append(
                            float(score)
                        )

                    except (
                        ValueError,
                        TypeError,
                    ):

                        pass

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Sources Retrieved",
                        len(sources),
                    )

                with col2:

                    if rankings:

                        st.metric(
                            "Best Ranking",
                            f"{max(rankings):.4f}",
                        )

                    else:

                        st.metric(
                            "Best Ranking",
                            "N/A",
                        )

                with col3:

                    if rankings:

                        st.metric(
                            "Average Ranking",
                            f"{sum(rankings) / len(rankings):.4f}",
                        )

                    else:

                        st.metric(
                            "Average Ranking",
                            "N/A",
                        )

                # ============================================
                # SOURCE DETAILS
                # ============================================

                for index, source in enumerate(
                    sources,
                    start=1,
                ):

                    page = source.get(
                        "page_number",
                        "Unknown",
                    )

                    chunk_id = source.get(
                        "chunk_id",
                        "Unknown",
                    )

                    ranking = source.get(
                        "ranking_score",
                        0.0,
                    )

                    with st.expander(
                        f"Source {index} — "
                        f"Page {page} — "
                        f"Chunk {chunk_id}"
                    ):

                        col1, col2 = st.columns(2)

                        with col1:

                            st.write(
                                f"**Page:** {page}"
                            )

                            st.write(
                                f"**Chunk ID:** "
                                f"{chunk_id}"
                            )

                        with col2:

                            st.write(
                                f"**Ranking Score:** "
                                f"{ranking:.4f}"
                            )

                        st.markdown(
                            "**Retrieved Text:**"
                        )

                        st.markdown(
                            source.get(
                                "text",
                                "",
                            )
                        )

            else:

                st.info(
                    "No supporting sources were retrieved."
                )

        except Exception as e:

            st.error(
                f"❌ Error while generating the answer: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ResearchLens • Grounded RAG Research Assistant"
)