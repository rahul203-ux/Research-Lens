from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.chunker import create_chunks
from ingestion.embedder import create_embeddings
from vectorstore.qdrant_client import upload_embeddings


PDF_PATH = "data/papers/research_paper.pdf"


print("=" * 70)
print("ResearchLens - Upload Pipeline")
print("=" * 70)


# --------------------------------------------------
# 1. Load PDF
# --------------------------------------------------

print("\n[1/4] Loading PDF...")

pages = extract_text_from_pdf(PDF_PATH)

print(f"Total pages: {len(pages)}")


# --------------------------------------------------
# 2. Create chunks
# --------------------------------------------------

print("\n[2/4] Creating chunks...")

chunks = create_chunks(pages)

print(f"Total chunks: {len(chunks)}")


# --------------------------------------------------
# 3. Generate embeddings
# --------------------------------------------------

print("\n[3/4] Generating embeddings...")

texts = [chunk["text"] for chunk in chunks]

embeddings = create_embeddings(texts)

print(f"Total embeddings: {len(embeddings)}")
print(f"Embedding dimensions: {embeddings.shape[1]}")


# --------------------------------------------------
# 4. Upload to Qdrant
# --------------------------------------------------

print("\n[4/4] Uploading embeddings to Qdrant...")

upload_embeddings(chunks, embeddings)


print("\n" + "=" * 70)
print("UPLOAD COMPLETED SUCCESSFULLY!")
print("=" * 70)