"""
ResearchLens - High Quality Retrieval
--------------------------------------

Pipeline:

Question
   ↓
SentenceTransformer embedding
   ↓
Qdrant candidate retrieval
   ↓
Remove bibliography / acknowledgement noise
   ↓
CrossEncoder reranking
   ↓
Question-term relevance
   ↓
Duplicate removal
   ↓
Evidence quality filtering
   ↓
Top relevant chunks
"""

from ingestion.embedder import create_embeddings
from vectorstore.qdrant_client import search_vectors
from sentence_transformers import CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INITIAL_K = 30
FINAL_K = 5

# Minimum number of words a useful evidence chunk should have.
MIN_TEXT_WORDS = 20


# ============================================================
# LOAD CROSS ENCODER
# ============================================================

reranker = CrossEncoder(RERANKER_MODEL)


# ============================================================
# EXCLUDED SECTIONS
# ============================================================

EXCLUDED_HEADINGS = {
    "references",
    "reference",
    "bibliography",
    "acknowledgement",
    "acknowledgements",
    "acknowledgment",
    "acknowledgments",
    "author biography",
    "authors biography",
    "conflict of interest",
    "declaration of interest",
    "declarations",
    "funding",
}


# ============================================================
# REFERENCE DETECTION
# ============================================================

def is_excluded_chunk(text):
    """
    Detect bibliography/reference/acknowledgement chunks.

    This deliberately avoids simply checking whether the word
    'reference' appears anywhere in a technical paragraph.
    """

    if not text:
        return True

    text_lower = text.lower().strip()

    if len(text_lower.split()) < MIN_TEXT_WORDS:
        return True

    # --------------------------------------------------------
    # Heading detection
    # --------------------------------------------------------

    lines = [
        line.strip().lower()
        for line in text.splitlines()
        if line.strip()
    ]

    for line in lines:

        cleaned = line.strip(" .:-")

        if cleaned in EXCLUDED_HEADINGS:
            return True

    # --------------------------------------------------------
    # Reference patterns
    # --------------------------------------------------------

    reference_patterns = [
        "[1]",
        "[2]",
        "[3]",
        "[4]",
        "[5]",
        "doi:",
        "https://doi.org/",
        "http://doi.org/",
    ]

    reference_hits = sum(
        pattern in text_lower
        for pattern in reference_patterns
    )

    # Several reference indicators = very likely bibliography.
    if reference_hits >= 2:
        return True

    # --------------------------------------------------------
    # Reference-page detection
    # --------------------------------------------------------

    first_250 = text_lower[:250]

    if "references" in first_250 and reference_hits >= 1:
        return True

    return False


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text for duplicate detection.
    """

    if not text:
        return ""

    return " ".join(
        text.lower().split()
    )


# ============================================================
# KEYWORD RELEVANCE
# ============================================================

STOP_WORDS = {
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "does",
    "do",
    "did",
    "can",
    "could",
    "would",
    "should",
    "this",
    "that",
    "these",
    "those",
    "about",
    "explain",
    "describe",
    "discuss",
    "tell",
    "give",
    "list",
    "paper",
    "article",
}


def tokenize(text):
    """
    Simple tokenization used only as a secondary signal.
    """

    words = set()

    for word in text.lower().split():

        cleaned = "".join(
            character
            for character in word
            if character.isalnum()
        )

        if (
            cleaned
            and cleaned not in STOP_WORDS
            and len(cleaned) > 1
        ):
            words.add(cleaned)

    return words


def calculate_keyword_overlap(question, text):
    """
    Secondary lexical relevance signal.

    CrossEncoder remains the primary ranking mechanism.
    """

    question_words = tokenize(question)
    text_words = tokenize(text)

    if not question_words:
        return 0.0

    overlap = question_words.intersection(
        text_words
    )

    return len(overlap) / len(question_words)


# ============================================================
# QUERY TYPE
# ============================================================

def detect_query_type(question):
    """
    Detect whether the user is asking for a particular
    paper section.

    This allows section-specific questions such as:

        abstract
        methodology
        conclusion
        limitations
        future scope
        results
    """

    q = question.lower()

    if "abstract" in q:
        return "abstract"

    if (
        "methodology" in q
        or "method" in q
        or "approach" in q
        or "proposed method" in q
    ):
        return "methodology"

    if (
        "conclusion" in q
        or "conclude" in q
    ):
        return "conclusion"

    if (
        "limitation" in q
        or "limitations" in q
    ):
        return "limitations"

    if (
        "future work" in q
        or "future scope" in q
        or "future enhancement" in q
    ):
        return "future"

    if (
        "result" in q
        or "results" in q
        or "performance" in q
    ):
        return "results"

    if (
        "introduction" in q
        or "background" in q
    ):
        return "introduction"

    return None


# ============================================================
# SECTION BONUS
# ============================================================

def section_bonus(question, text):
    """
    Small bonus for section-specific queries.

    This is deliberately small so that section words do not
    overpower semantic relevance.
    """

    query_type = detect_query_type(question)

    if not query_type:
        return 0.0

    text_lower = text.lower()

    if query_type == "abstract":

        if (
            "abstract" in text_lower
            or "we present" in text_lower
            or "this paper" in text_lower
            or "this work" in text_lower
        ):
            return 0.15

    elif query_type == "methodology":

        if (
            "methodology" in text_lower
            or "method" in text_lower
            or "proposed approach" in text_lower
            or "architecture" in text_lower
        ):
            return 0.15

    elif query_type == "conclusion":

        if "conclusion" in text_lower:
            return 0.15

    elif query_type == "limitations":

        if (
            "limitation" in text_lower
            or "limitations" in text_lower
        ):
            return 0.15

    elif query_type == "future":

        if (
            "future work" in text_lower
            or "future scope" in text_lower
            or "future" in text_lower
        ):
            return 0.15

    elif query_type == "results":

        if (
            "results" in text_lower
            or "performance" in text_lower
            or "evaluation" in text_lower
        ):
            return 0.15

    elif query_type == "introduction":

        if (
            "introduction" in text_lower
            or "background" in text_lower
        ):
            return 0.15

    return 0.0


# ============================================================
# MAIN RETRIEVAL
# ============================================================

def search_similar_chunks(
    question,
    top_k=FINAL_K,
    initial_k=INITIAL_K,
):
    """
    Retrieve and rerank evidence from Qdrant.
    """

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not question or not question.strip():
        return []

    question = question.strip()

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = create_embeddings(
        [question]
    )[0]

    # --------------------------------------------------------
    # Qdrant retrieval
    # --------------------------------------------------------

    results = search_vectors(
        query_embedding,
        top_k=initial_k,
    )

    if not results:
        return []

    # --------------------------------------------------------
    # Convert + clean
    # --------------------------------------------------------

    candidates = []

    seen_text = set()

    for result in results:

        payload = result.payload or {}

        text = payload.get(
            "text",
            ""
        )

        if not text:
            continue

        if is_excluded_chunk(text):
            continue

        normalized = normalize_text(text)

        if normalized in seen_text:
            continue

        seen_text.add(normalized)

        candidates.append(
            {
                "chunk_id": payload.get(
                    "chunk_id"
                ),
                "page_number": payload.get(
                    "page_number"
                ),
                "text": text,
                "qdrant_score": float(
                    getattr(
                        result,
                        "score",
                        0.0,
                    )
                ),
            }
        )

    if not candidates:
        return []

    # --------------------------------------------------------
    # CrossEncoder reranking
    # --------------------------------------------------------

    pairs = [
        [
            question,
            candidate["text"],
        ]
        for candidate in candidates
    ]

    rerank_scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    for candidate, score in zip(
        candidates,
        rerank_scores,
    ):

        candidate["rerank_score"] = float(
            score
        )

    # --------------------------------------------------------
    # Calculate secondary signals
    # --------------------------------------------------------

    for candidate in candidates:

        candidate["keyword_score"] = (
            calculate_keyword_overlap(
                question,
                candidate["text"],
            )
        )

        candidate["section_bonus"] = (
            section_bonus(
                question,
                candidate["text"],
            )
        )

    # --------------------------------------------------------
    # Normalize CrossEncoder scores
    #
    # IMPORTANT:
    # This is ONLY for ranking.
    # It must NOT be displayed as similarity.
    # --------------------------------------------------------

    rerank_values = [
        candidate["rerank_score"]
        for candidate in candidates
    ]

    minimum = min(rerank_values)
    maximum = max(rerank_values)

    score_range = maximum - minimum

    for candidate in candidates:

        if score_range == 0:

            candidate["normalized_rerank"] = 1.0

        else:

            candidate["normalized_rerank"] = (
                candidate["rerank_score"]
                - minimum
            ) / score_range

    # --------------------------------------------------------
    # Final ranking
    #
    # CrossEncoder = dominant signal
    # Keyword = secondary
    # Section = small bonus
    # --------------------------------------------------------

    for candidate in candidates:

        candidate["ranking_score"] = (
            0.80
            * candidate["normalized_rerank"]
            +
            0.15
            * candidate["keyword_score"]
            +
            0.05
            * candidate["section_bonus"]
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: item["ranking_score"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Select results
    #
    # No arbitrary CrossEncoder >= 0 threshold.
    # --------------------------------------------------------

    selected = candidates[:top_k]

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    final_results = []

    for candidate in selected:

        final_results.append(
            {
                "chunk_id": candidate[
                    "chunk_id"
                ],

                "page_number": candidate[
                    "page_number"
                ],

                "text": candidate[
                    "text"
                ],

                # REAL QDRANT VECTOR SCORE
                "qdrant_score": candidate[
                    "qdrant_score"
                ],

                # REAL CROSSENCODER SCORE
                "rerank_score": candidate[
                    "rerank_score"
                ],

                # INTERNAL RANKING ONLY
                "ranking_score": candidate[
                    "ranking_score"
                ],

                # Backward compatibility
                "score": candidate[
                    "qdrant_score"
                ],
            }
        )

    return final_results


# ============================================================
# COMPATIBILITY WRAPPER
# ============================================================

def retrieve_chunks(
    question,
    top_k=FINAL_K,
):
    """
    Used by services/chat_service.py.
    """

    return search_similar_chunks(
        question=question,
        top_k=top_k,
        initial_k=INITIAL_K,
    )