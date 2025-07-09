from query_langchain_pgvector import find_and_generate_answer

if __name__ == "__main__":
    # 这里随便写一个你想问的问题
    question = "What is fine-tuning in transformers?"
    result = find_and_generate_answer(question, top_k=5)

    print("Answer:\n", result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source['source']} | Score: {source['relevance_score']}")
        print(f"  Snippet: {source['snippet']}\n")
