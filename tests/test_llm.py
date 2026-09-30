from generation.llm import generate_answer


prompt = """
Explain in simple terms what a 5-stage RISC-V pipeline is.
"""


print("=" * 70)
print("ResearchLens - LLM Test")
print("=" * 70)

answer = generate_answer(prompt)

print("\nLLM Response:")
print(answer)