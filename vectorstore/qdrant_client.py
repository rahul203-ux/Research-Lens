import os

from dotenv import load_dotenv

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# ============================================================
# QDRANT CONFIGURATION
# ============================================================

COLLECTION_NAME = "research_papers"

VECTOR_SIZE = 384


# ============================================================
# GET QDRANT CLIENT
# ============================================================

def get_qdrant_client():
    """
    Create and return a connection to Qdrant Cloud.
    """

    if not QDRANT_URL:
        raise ValueError(
            "QDRANT_URL is missing from environment variables."
        )

    if not QDRANT_API_KEY:
        raise ValueError(
            "QDRANT_API_KEY is missing from environment variables."
        )

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    return client


# ============================================================
# CREATE COLLECTION
# ============================================================

def create_collection():
    """
    Create the ResearchLens collection
    if it does not already exist.
    """

    client = get_qdrant_client()

    existing_collections = [
        collection.name
        for collection
        in client.get_collections().collections
    ]

    if COLLECTION_NAME in existing_collections:

        print(
            f"Collection '{COLLECTION_NAME}' already exists."
        )

        return

    client.create_collection(

        collection_name=COLLECTION_NAME,

        vectors_config=VectorParams(

            size=VECTOR_SIZE,

            distance=Distance.COSINE
        )
    )

    print(
        f"Collection '{COLLECTION_NAME}' "
        "created successfully."
    )


# ============================================================
# CLEAR PREVIOUS DOCUMENT
# ============================================================

def clear_collection():
    """
    Delete the existing ResearchLens collection.

    ResearchLens follows a single-document workflow.
    When a new PDF is uploaded, the previous document's
    vectors must be removed so that retrieval cannot return
    information from an older document.
    """

    client = get_qdrant_client()

    existing_collections = [
        collection.name
        for collection
        in client.get_collections().collections
    ]

    if COLLECTION_NAME in existing_collections:

        client.delete_collection(
            collection_name=COLLECTION_NAME
        )

        print(
            f"Previous collection '{COLLECTION_NAME}' "
            "deleted successfully."
        )

    else:

        print(
            f"Collection '{COLLECTION_NAME}' "
            "does not exist. Nothing to clear."
        )


# ============================================================
# UPLOAD EMBEDDINGS
# ============================================================

def upload_embeddings(chunks, embeddings):
    """
    Upload text chunks and their embeddings to Qdrant.

    Each Qdrant point contains:

        ID
        Vector
        chunk_id
        page_number
        text
    """

    # Make sure collection exists
    create_collection()

    client = get_qdrant_client()

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if len(chunks) != len(embeddings):

        raise ValueError(

            f"Number of chunks ({len(chunks)}) "

            f"does not match number of embeddings "

            f"({len(embeddings)})."
        )

    # --------------------------------------------------------
    # Prepare Qdrant points
    # --------------------------------------------------------

    points = []

    for index, (chunk, embedding) in enumerate(

        zip(chunks, embeddings),

        start=1
    ):

        point = PointStruct(

            id=index,

            vector=embedding.tolist(),

            payload={

                "chunk_id": chunk["chunk_id"],

                "page_number": chunk["page_number"],

                "text": chunk["text"]
            }
        )

        points.append(point)

    # --------------------------------------------------------
    # Upload to Qdrant
    # --------------------------------------------------------

    client.upsert(

        collection_name=COLLECTION_NAME,

        points=points
    )

    print(
        f"Successfully uploaded "
        f"{len(points)} embeddings to Qdrant."
    )


# ============================================================
# SEARCH VECTORS
# ============================================================

def search_vectors(query_embedding, top_k=5):
    """
    Search Qdrant for the most relevant chunks.

    Args:
        query_embedding:
            Embedding vector of the user's question.

        top_k:
            Number of results to return.

    Returns:
        List of Qdrant search results.
    """

    client = get_qdrant_client()

    # --------------------------------------------------------
    # Search Qdrant
    # --------------------------------------------------------

    results = client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_embedding.tolist(),

        limit=top_k,

        with_payload=True
    )

    return results.points