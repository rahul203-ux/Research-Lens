from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.chunker import create_chunks


pdf_path = "data/papers/research_paper.pdf"

pages = extract_text_from_pdf(pdf_path)

print("Total pages:", len(pages))

chunks = create_chunks(
    pages,
    chunk_size=1000,
    chunk_overlap=200
)

print("Total chunks:", len(chunks))

for chunk in chunks:
    print("\n" + "=" * 80)
    print("Chunk ID:", chunk["chunk_id"])
    print("Page:", chunk["page_number"])
    print("=" * 80)
    print(chunk["text"])