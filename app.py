
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

from vectorstore.qdrant_client import (
    get_qdrant_client,
    COLLECTION_NAME,
    create_collection,
    upload_embeddings,
)


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


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = Path("data") / "uploads"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "paper_info" not in st.session_state:
    st.session_state.paper_info = None

if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None


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

    st.write(
        f"**LLM Model:** `{MODEL_NAME}`"
    )

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
        "ResearchLens processes one uploaded research paper "
        "at a time. When a new paper is uploaded, the previous "
        "paper's vectors are removed from Qdrant."
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
    help="Upload one PDF research paper.",
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    new_file = (
        st.session_state.current_file_name
        != uploaded_file.name
    )

    if new_file:

        try:

            # ====================================================
            # SAVE PDF
            # ====================================================

            file_path = UPLOAD_DIR / uploaded_file.name

            with open(
                file_path,
                "wb",
            ) as f:

                f.write(
                    uploaded_file.getbuffer()
                )

            file_size_mb = (
                file_path.stat().st_size
                / (1024 * 1024)
            )

            # ====================================================
            # PROCESSING STATUS
            # ====================================================

            with st.status(
                "🔄 Processing research paper...",
                expanded=True,
            ) as status:

                # ====================================================
                # REMOVE PREVIOUS QDRANT COLLECTION
                # ====================================================

                st.write(
                    "🗑️ Removing previous paper vectors..."
                )

                client = get_qdrant_client()

                try:

                    client.delete_collection(
                        collection_name=COLLECTION_NAME
                    )

                    st.write(
                        "✓ Previous paper data removed."
                    )

                except Exception as e:

                    error_text = str(e).lower()

                    if (
                        "not found" in error_text
                        or "does not exist" in error_text
                        or "404" in error_text
                    ):

                        st.write(
                            "✓ No previous collection found."
                        )

                    else:

                        raise e

                # ====================================================
                # CREATE FRESH COLLECTION
                # ====================================================

                st.write(
                    "🗄️ Creating fresh Qdrant collection..."
                )

                create_collection()

                st.write(
                    "✓ Fresh Qdrant collection ready."
                )

                # ====================================================
                # STEP 1 - TEXT EXTRACTION
                # ====================================================

                st.write(
                    "📝 Extracting text from PDF..."
                )

                pages = extract_text_from_pdf(
                    str(file_path)
                )

                page_count = len(pages)

                total_characters = sum(
                    len(
                        page.get(
                            "text",
                            "",
                        )
                    )
                    for page in pages
                )

                st.write(
                    f"✓ Extracted **{page_count} pages**"
                )

                st.write(
                    f"✓ Extracted **{total_characters:,} characters**"
                )

                # ====================================================
                # STEP 2 - CHUNKING
                # ====================================================

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

                # ====================================================
                # STEP 3 - EMBEDDINGS
                # ====================================================

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

                    if len(
                        embeddings.shape
                    ) > 1:

                        embedding_dimension = (
                            embeddings.shape[1]
                        )

                    else:

                        embedding_dimension = 0

                else:

                    embeddings = []

                    embedding_count = 0

                    embedding_dimension = 0

                st.write(
                    f"✓ Generated **{embedding_count} embeddings**"
                )

                st.write(
                    f"✓ Vector dimension: "
                    f"**{embedding_dimension}**"
                )

                # ====================================================
                # STEP 4 - QDRANT
                # ====================================================

                if (
                    chunks
                    and embedding_count > 0
                ):

                    st.write(
                        "🗄️ Uploading current paper vectors to Qdrant..."
                    )

                    upload_embeddings(
                        chunks,
                        embeddings,
                    )

                    st.write(
                        f"✓ Stored **{embedding_count} vectors** "
                        "for the current paper."
                    )

                else:

                    raise ValueError(
                        "No chunks or embeddings were generated "
                        "from the uploaded PDF."
                    )

                # ====================================================
                # PROCESSING COMPLETE
                # ====================================================

                status.update(
                    label=(
                        "✅ Research paper processing completed!"
                    ),
                    state="complete",
                )

            # ========================================================
            # SAVE SESSION INFORMATION
            # ========================================================

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

            st.session_state.current_file_name = (
                uploaded_file.name
            )

            st.session_state.processing_done = True

            st.success(
                "Research paper uploaded and processed: "
                f"**{uploaded_file.name}**"
            )

        except Exception as e:

            st.session_state.paper_info = None

            st.session_state.processing_done = False

            st.error(
                f"❌ Error while processing PDF: {e}"
            )

    else:

        st.success(
            "Current research paper: "
            f"**{uploaded_file.name}**"
        )


# ============================================================
# PAPER PROCESSING INFORMATION
# ============================================================

if st.session_state.paper_info is not None:

    info = st.session_state.paper_info

    st.divider()

    st.header(
        "📊 Research Paper Processing"
    )

    # ========================================================
    # MAIN METRICS
    # ========================================================

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

    # ========================================================
    # PROCESSING DETAILS
    # ========================================================

    st.markdown(
        "### 🔍 Processing Details"
    )

    detail_cols = st.columns(5)

    with detail_cols[0]:

        st.success(
            "📄\n\n**PDF Upload**"
        )

    with detail_cols[1]:

        st.success(
            "📝\n\n**Text Extraction**"
        )

    with detail_cols[2]:

        st.success(
            "✂️\n\n**Chunking**"
        )

    with detail_cols[3]:

        st.success(
            "🔢\n\n**Embeddings**"
        )

    with detail_cols[4]:

        st.success(
            "🗄️\n\n**Qdrant**"
        )


# ============================================================
# CURRENT PAPER
# ============================================================

if st.session_state.paper_info is not None:

    st.divider()

    st.subheader(
        "📚 Current Research Paper"
    )

    st.write(
        "📄 **"
        f"{st.session_state.paper_info['file_name']}"
        "**"
    )

    st.caption(
        "Only this paper is currently stored in "
        "the Qdrant collection and used for "
        "question answering."
    )


# ============================================================
# QUESTION SECTION
# ============================================================

st.divider()

st.header(
    "💬 Ask a Question"
)

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

    elif st.session_state.paper_info is None:

        st.warning(
            "Please upload and process a research paper first."
        )

    else:

        try:

            with st.spinner(
                "🔍 Searching the uploaded paper..."
            ):

                result = answer_question(
                    question.strip(),
                    top_k=MAX_SOURCES,
                )

            # ====================================================
            # ANSWER
            # ====================================================

            st.markdown(
                "## 💡 Answer"
            )

            answer = result.get(
                "answer",
                "No answer generated.",
            )

            st.markdown(answer)

            # ====================================================
            # SOURCES
            # ====================================================

            sources = result.get(
                "sources",
                [],
            )

            if sources:

                st.markdown(
                    "## 📚 Retrieved Sources"
                )

                # ====================================================
                # SOURCE STATISTICS
                # ====================================================

                rankings = [
                    source.get(
                        "ranking_score",
                        0.0,
                    )
                    for source in sources
                ]

                best_ranking = max(
                    rankings
                )

                average_ranking = (
                    sum(rankings)
                    / len(rankings)
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Sources Retrieved",
                        len(sources),
                    )

                with col2:

                    st.metric(
                        "Best Ranking",
                        f"{best_ranking:.4f}",
                    )

                with col3:

                    st.metric(
                        "Average Ranking",
                        f"{average_ranking:.4f}",
                    )

                # ====================================================
                # SOURCE DETAILS
                # ====================================================

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

                        st.write(
                            f"**Page:** {page}"
                        )

                        st.write(
                            f"**Chunk ID:** {chunk_id}"
                        )

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
