from services.chat_service import answer_question


print("=" * 80)
print("ResearchLens - RAG Test")
print("=" * 80)


question = "What are the main challenges of the proposed processor?"


print("\nQUESTION:")
print(question)


result = answer_question(question)


print("\nANSWER:")
print(result["answer"])


print("\n" + "=" * 80)
print("SOURCES")
print("=" * 80)


for source in result["sources"]:

    print(
        f"\nChunk ID: {source['chunk_id']}"
        f" | Page: {source['page_number']}"
        f" | Score: {source['score']:.4f}"
    )

    print("-" * 80)

    print(source["text"][:500])