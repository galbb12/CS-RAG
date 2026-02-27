from rag_engine import RAGEngine


def main():
    engine = RAGEngine()
    messages: list[dict] = []

    print("HAIFA-RAG Bot Ready! (type 'exit' to quit)\n")

    while True:
        question = input("שאלה: ").strip()
        if not question or question == "exit":
            break

        messages.append({"role": "user", "content": question})
        result = engine.query(messages)
        messages.append({"role": "assistant", "content": result.content})

        print(f"\n{result.content}\n")


if __name__ == "__main__":
    main()
