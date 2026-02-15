import json
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI

from metadata_match import MetadataMatch
from comment_match_prompt import CommentMatchPrompt

load_dotenv()

SYSTEM_PROMPT = """אתה עוזר לסטודנטים בחוג למדעי המחשב באוניברסיטת חיפה.
אתה עונה על שאלות לגבי קורסים, מרצים, מבחנים ועוד על סמך ביקורות אמיתיות של סטודנטים.

כללים:
- ענה רק על סמך הביקורות שסופקו לך. אל תמציא מידע.
- אם אין מספיק מידע בביקורות, אמור זאת בכנות.
- ציין מגמות חוזרות בביקורות (למשל אם כמה סטודנטים מסכימים על נקודה מסוימת).
- השתמש בעברית בתשובות שלך.
- היה תמציתי ועניני.
"""


def format_context(docs: list[Document]) -> str:
    """Format retrieved documents into a context string for the LLM."""
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        header = f"ביקורת {i}"
        if meta.get("course_name"):
            header += f" | קורס: {meta['course_name']}"
        if meta.get("lecturer"):
            header += f" | מרצה: {meta['lecturer']}"
        if meta.get("course_type"):
            header += f" | סוג: {meta['course_type']}"
        if meta.get("credit_points"):
            header += f" | נ\"ז: {meta['credit_points']}"
        parts.append(f"[{header}]\n{doc.page_content}")
    return "\n\n".join(parts)


def main():
    # Load documents
    with open("documents.json", encoding="utf-8") as f:
        raw_docs = json.load(f)
    documents = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in raw_docs]

    # Init components
    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.environ["OPENAI_API_KEY"],
    )

    metadata_matcher = MetadataMatch(documents, llm)
    comment_matcher = CommentMatchPrompt(embeddings, documents, k=10)

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("CS-RAG Bot Ready! (type 'exit' to quit)\n")

    while True:
        question = input("שאלה: ").strip()
        if not question or question == "exit":
            break

        # Step 1: extract metadata filters
        filters = metadata_matcher.extract_filters(question)
        metadata_filter = {k: v for k, v in filters.model_dump().items() if k != "query" and v is not None}

        # Step 2: semantic search with filters
        top_docs = comment_matcher.run(filters.query, metadata_filter)

        # Step 3: build the user message with context
        context = format_context(top_docs)
        user_message = f"ביקורות רלוונטיות:\n{context}\n\nשאלת הסטודנט: {question}"

        conversation.append({"role": "user", "content": user_message})

        # Step 4: LLM answers with context
        response = llm.invoke(conversation)
        answer = response.content

        conversation.append({"role": "assistant", "content": answer})

        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
