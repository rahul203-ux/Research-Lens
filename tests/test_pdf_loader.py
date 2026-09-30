from ingestion.pdf_loader import extract_text_from_pdf


pdf_path = "data/papers/research_paper.pdf"

pages = extract_text_from_pdf(pdf_path)

print("Total pages:", len(pages))

for page in pages:
    print("\n" + "=" * 80)
    print("Page:", page["page_number"])
    print("=" * 80)
    print(page["text"])