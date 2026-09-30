"""
ResearchLens - Grounded RAG Research Paper Assistant
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from services.chat_service import answer_question


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
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = Path("data") / "uploads"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

st.title("🔬 ResearchLens")

st.subheader(
    "Grounded Research Paper Question Answering"
)

st.write(
    "Upload a research paper and ask questions "
    "using retrieval-augmented generation."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write(
        f"**Model:** `{MODEL_NAME}`"
    )

    st.write(
        "**Maximum sources:** `5`"
    )

    st.write(
        "**Minimum ranking:** `0.3`"
    )

    st.divider()

    st.info(
        "ResearchLens retrieves relevant sections from "
        "the research paper and uses an LLM to generate "
        "a grounded answer."
    )


# ============================================================
# UPLOAD SECTION
# ============================================================

st.header("📄 Upload Research Paper")

uploaded_file = st.file_uploader(
    "Choose a research paper",
    type=["pdf"],
    help="Upload a PDF research paper."
)


# ============================================================
# HANDLE UPLOAD
# ============================================================

if uploaded_file is not None:

    file_path = UPLOAD_DIR / uploaded_file.name

    try:

        with open(
            file_path,
            "wb"
        ) as f:

            f.write(
                uploaded_file.getbuffer()
            )

        st.success(
            f"Research paper uploaded: "
            f"**{uploaded_file.name}**"
        )

        st.info(
            "The PDF has been saved successfully. "
            "It must now be passed through the ingestion "
            "and vector-store pipeline before questions "
            "can use this new paper."
        )

        st.write(
            f"**File:** `{file_path}`"
        )

    except Exception as e:

        st.error(
            f"Could not save the uploaded file: {e}"
        )


# ============================================================
# CURRENT UPLOADED FILE
# ============================================================

uploaded_files = list(
    UPLOAD_DIR.glob("*.pdf")
)

if uploaded_files:

    st.divider()

    st.subheader("📚 Uploaded Papers")

    for pdf in uploaded_files:

        st.write(
            f"📄 {pdf.name}"
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
    height=120
)


# ============================================================
# ASK BUTTON
# ============================================================

ask_button = st.button(
    "🔍 Ask ResearchLens",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANSWER
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching the research paper..."
        ):

            result = answer_question(
                question
            )

        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        st.markdown(
            "## 💡 Answer"
        )

        st.markdown(
            result.get(
                "answer",
                "No answer generated."
            )
        )

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        sources = result.get(
            "sources",
            []
        )

        if sources:

            st.markdown(
                "## 📚 Retrieved Sources"
            )

            for index, source in enumerate(
                sources,
                start=1
            ):

                page = source.get(
                    "page_number",
                    "Unknown"
                )

                chunk_id = source.get(
                    "chunk_id",
                    "Unknown"
                )

                ranking = source.get(
                    "ranking_score",
                    0.0
                )

                with st.expander(
                    f"Source {index} — Page {page}"
                ):

                    st.write(
                        f"**Chunk ID:** {chunk_id}"
                    )

                    st.write(
                        f"**Ranking Score:** "
                        f"{ranking:.4f}"
                    )

                    st.markdown(
                        source.get(
                            "text",
                            ""
                        )
                    )

        else:

            st.info(
                "No supporting sources were retrieved."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ResearchLens • Grounded RAG Research Assistant"
)