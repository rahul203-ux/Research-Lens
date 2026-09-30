"""
ResearchLens - Grounded RAG Answer Generation
"""

import os

from groq import Groq

from retrieval.retriever import retrieve_chunks


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv(
    "GROQ_MODEL",
     "openai/gpt-oss-120b"
)

MAX_CONTEXT_CHUNKS = 5

# Minimum ranking score required before allowing an answer.
# This is deliberately conservative.
MIN_CONTEXT_RANKING = 0.30


# ============================================================
# GROQ CLIENT
# ============================================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is missing from the .env file."
    )

client = Groq(
    api_key=api_key
)


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(chunks):
    """
    Build a clearly labelled evidence context.
    """

    context_parts = []

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {index}
Chunk ID: {chunk.get("chunk_id", "Unknown")}
Page: {chunk.get("page_number", "Unknown")}

TEXT:
{chunk.get("text", "")}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# ANSWER QUESTION
# ============================================================

def answer_question(
    question,
    top_k=MAX_CONTEXT_CHUNKS,
):
    """
    Retrieve evidence and generate a grounded answer.
    """

    if not question or not question.strip():

        return {
            "answer": "Please enter a question.",
            "sources": [],
        }

    question = question.strip()

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    chunks = retrieve_chunks(
        question,
        top_k=top_k,
    )

    if not chunks:

        return {
            "answer": (
                "The uploaded research paper does not "
                "contain enough retrievable information "
                "to answer this question reliably."
            ),
            "sources": [],
        }

    # --------------------------------------------------------
    # FILTER VERY WEAK EVIDENCE
    #
    # ranking_score is an internal ranking value, not
    # Qdrant similarity.
    # --------------------------------------------------------

    usable_chunks = [
        chunk
        for chunk in chunks
        if chunk.get(
            "ranking_score",
            0.0
        ) >= MIN_CONTEXT_RANKING
    ]

    # If everything is weak, do not hallucinate.
    if not usable_chunks:

        return {
            "answer": (
                "The retrieved sections of the paper "
                "do not provide enough information to "
                "answer this question reliably."
            ),
            "sources": [],
        }

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    usable_chunks = usable_chunks[
        :MAX_CONTEXT_CHUNKS
    ]

    context = build_context(
        usable_chunks
    )

    # --------------------------------------------------------
    # STRICT SYSTEM PROMPT
    # --------------------------------------------------------

    system_prompt = """
You are ResearchLens, a research-paper question answering assistant.

Your ONLY source of factual information is the provided paper context.

STRICT RULES:

1. Answer only from the supplied context.
2. Do not use outside knowledge.
3. Do not invent facts, numbers, results, methods, authors,
   limitations, conclusions, or claims.
4. If the context does not contain enough information,
   explicitly say that the paper context does not provide
   enough information.
5. Never treat the source metadata or chunk IDs as facts.
6. Combine multiple source chunks when necessary.
7. Give a concise but complete answer.
8. When making a factual claim, cite the source using:
   [Source 1], [Source 2], etc.
9. Do not cite a source unless that source actually supports
   the claim.
10. If the user asks for the abstract, summarize only the
    retrieved abstract/introduction content. Do not fabricate
    an abstract.
11. Do not answer a question merely because the retrieved text
    contains related words. The evidence must actually support
    the answer.
"""


    user_prompt = f"""
QUESTION:
{question}

PAPER CONTEXT:
{context}

Answer the question using ONLY the PAPER CONTEXT.

Every important factual statement should have an appropriate
source citation such as [Source 1].
"""


    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        max_tokens=1000,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    answer = response.choices[0].message.content

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {
        "answer": answer,
        "sources": usable_chunks,
    }