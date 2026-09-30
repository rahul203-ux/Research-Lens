from retrieval.retriever import search_similar_chunks


query = "What are the main challenges of the proposed processor?"


print("=" * 80)
print("ResearchLens - Retrieval Test")
print("=" * 80)

print(f"\nQuestion: {query}")

results = search_similar_chunks(query, top_k=5)

print(f"\nRetrieved chunks: {len(results)}")


for i, result in enumerate(results, start=1):

    print("\n" + "-" * 80)

    print(f"Result: {i}")
    print(f"Similarity Score: {result['score']:.4f}")
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Page: {result['page_number']}")

    print("\nText:")
    print(result["text"])