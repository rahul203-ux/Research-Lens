from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.chunker import create_chunks
from ingestion.embedder import create_embeddings


pdf_path = "data/papers/research_paper.pdf"


# Extract PDF
pages = extract_text_from_pdf(pdf_path)

print("Total pages:", len(pages))


# Create chunks
chunks = create_chunks(
    pages,
    chunk_size=1000,
    chunk_overlap=200
)

print("Total chunks:", len(chunks))


# Extract text from chunks
texts = [chunk["text"] for chunk in chunks]


# Create embeddings
embeddings = create_embeddings(texts)


print("\nEmbedding generation completed.")
print("Number of embeddings:", len(embeddings))
print("Embedding dimensions:", embeddings.shape[1])