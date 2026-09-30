import fitz


def extract_text_from_pdf(pdf_path):
    """
    Extract text from every page of a PDF.

    Returns:
        list: Each item contains page number and page text.
    """

    pages = []

    document = fitz.open(pdf_path)

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        pages.append({
            "page_number": page_number,
            "text": text.strip()
        })

    document.close()

    return pages