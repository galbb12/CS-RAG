import time

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


FAISS_INDEX_PATH = "faiss_index"


class CommentMatchPrompt:
    """Pre-embeds all documents once. At query time, filters by metadata and returns top-k."""

    def __init__(self, embeddings: GoogleGenerativeAIEmbeddings, documents: list[Document], k: int = 10):
        self.embeddings = embeddings
        self.k = k
        self.vectorstore = self._load_or_build_index(documents)

    def _load_or_build_index(self, documents: list[Document]) -> FAISS:
        """Load saved FAISS index, or build + save it from scratch."""
        try:
            vs = FAISS.load_local(FAISS_INDEX_PATH, self.embeddings, allow_dangerous_deserialization=True)
            print(f"Loaded existing FAISS index from {FAISS_INDEX_PATH}")
            return vs
        except Exception:
            print(f"Building FAISS index for {len(documents)} documents...")
            # Batch to respect rate limits (100 req/min on free tier)
            batch_size = 80
            vs = None
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                print(f"  Embedding batch {i // batch_size + 1} ({len(batch)} docs)...")
                if vs is None:
                    vs = FAISS.from_documents(batch, self.embeddings)
                else:
                    vs.add_documents(batch)
                if i + batch_size < len(documents):
                    time.sleep(61)  # wait for rate limit window to reset
            vs.save_local(FAISS_INDEX_PATH)
            print(f"Saved FAISS index to {FAISS_INDEX_PATH}")
            return vs

    def run(self, query: str, metadata_filter: dict) -> list[Document]:
        """Semantic search with metadata filtering. Only embeds the query (1 API call)."""
        clean_filter = {k: v for k, v in metadata_filter.items() if v is not None}

        # fetch_k must be large enough so filtered results aren't missed.
        # FAISS first fetches fetch_k nearest neighbors, then filters by metadata.
        fetch_k = len(self.vectorstore.docstore._dict)

        if clean_filter:
            results = self.vectorstore.similarity_search(query, k=self.k, filter=clean_filter, fetch_k=fetch_k)
        else:
            results = self.vectorstore.similarity_search(query, k=self.k)
        return results


# ---- Demo ----
if __name__ == "__main__":
    import json
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from metadata_match import MetadataMatch

    load_dotenv()

    # Load documents
    with open("documents.json", encoding="utf-8") as f:
        raw_docs = json.load(f)
    documents = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in raw_docs]

    # Init components
    llm = ChatOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-2.5-flash",
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.environ["OPENAI_API_KEY"],
    )

    metadata_matcher = MetadataMatch(documents, llm)
    comment_matcher = CommentMatchPrompt(embeddings, documents, k=10)  # builds index once

    # Test queries
    examples = [
        "מה אומרים על אור דונקלמן?",
        "קורסי בחירה קלים",
        "איך המבחן במערכות הפעלה?",
    ]

    for question in examples:
        print(f"\n{'='*60}")
        print(f"Question: {question}")

        # Step 1: LLM extracts metadata filters
        filters = metadata_matcher.extract_filters(question)
        print(f"Filters: {filters}")

        # Step 2: semantic search with metadata filter (1 embed call for the query)
        metadata_filter = {k: v for k, v in filters.model_dump().items() if k != "query" and v is not None}
        top_comments = comment_matcher.run(filters.query, metadata_filter)
        print(f"Top {len(top_comments)} comments:\n")

        for i, doc in enumerate(top_comments, 1):
            meta = doc.metadata
            print(f"  {i}. [{meta.get('course_name', '')} | {meta.get('lecturer', '')}]")
            print(f"     {doc.page_content}")
            print()
