def create_chunks(pages, chunk_size=1000, chunk_overlap=200):
    """
    Split page-level text into smaller overlapping chunks.

    Args:
        pages: List of dictionaries containing page number and text.
        chunk_size: Maximum approximate number of characters per chunk.
        chunk_overlap: Number of characters shared between chunks.

    Returns:
        List of chunk dictionaries.
    """

    chunks = []

    chunk_id = 1

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]

        start = 0

        while start < len(text):
            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "page_number": page_number,
                    "text": chunk_text
                })

                chunk_id += 1

            start = end - chunk_overlap

    return chunks