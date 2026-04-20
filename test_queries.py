from src.qa.rag_pipeline import RAGPipeline

queries = [
    "What are the main barriers to telemedicine adoption in rural areas?",
    "How do mHealth apps improve diabetes management?",
    "Can digital health education reduce healthcare disparities?"
]

print("🚀 Starting Diverse Query Testing\n")
bot = RAGPipeline()

for i, query in enumerate(queries, 1):
    print(f"========================================")
    print(f"QUERY {i}: {query}")
    print(f"========================================")
    answer = bot.ask_question(query)
    print("\nANSWER:")
    print(answer)
    print("\n\n")
