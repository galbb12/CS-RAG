import os
import time
from typing import Optional

import faiss as faiss_lib
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index"))


class CommentMatchPrompt:
    """Pre-embeds all documents once. Exposes a LangChain tool for the LLM to search."""

    def __init__(self, embeddings: GoogleGenerativeAIEmbeddings, documents: list[Document], k: int = 10):
        self.embeddings = embeddings
        self.k = k
        self.documents = documents
        self.vectorstore = self._load_or_build_index(documents)

        self.all_lecturers = sorted(set(d.metadata["lecturer"] for d in documents if d.metadata.get("lecturer")))
        self.all_courses = sorted(set(d.metadata["course_name"] for d in documents if d.metadata.get("course_name")))
        self.all_types = sorted(set(d.metadata["course_type"] for d in documents if d.metadata.get("course_type")))

    def _build_docstore(self, documents: list[Document]) -> tuple[InMemoryDocstore, dict[int, str]]:
        """Build a docstore + index-to-id mapping from documents."""
        docstore_dict = {}
        index_to_id = {}
        for i, doc in enumerate(documents):
            doc_id = str(i)
            docstore_dict[doc_id] = doc
            index_to_id[i] = doc_id
        return InMemoryDocstore(docstore_dict), index_to_id

    def _load_or_build_index(self, documents: list[Document]) -> FAISS:
        """Load saved FAISS vectors or embed from scratch. Docstore is always rebuilt from SQL."""
        faiss_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")

        if os.path.exists(faiss_file):
            print(f"Loaded FAISS vectors from {faiss_file}")
            index = faiss_lib.read_index(faiss_file)
            docstore, index_to_id = self._build_docstore(documents)
            return FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_id,
            )

        print(f"Building FAISS index for {len(documents)} documents...")
        batch_size = 80
        vs = None
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = i // batch_size + 1
            print(f"  Embedding batch {batch_num} ({len(batch)} docs)...")
            while True:
                try:
                    if vs is None:
                        vs = FAISS.from_documents(batch, self.embeddings)
                    else:
                        vs.add_documents(batch)
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        print(f"  Rate limited, waiting 60s...")
                        time.sleep(60)
                    else:
                        raise
            if i + batch_size < len(documents):
                time.sleep(61)

        # Save only the FAISS vectors
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        faiss_lib.write_index(vs.index, faiss_file)
        print(f"Saved FAISS vectors to {faiss_file}")

        # Build docstore so IDs are consistent with future loads
        docstore, index_to_id = self._build_docstore(documents)
        vs.docstore = docstore
        vs.index_to_docstore_id = index_to_id
        return vs

    def _random_docs(self, metadata_filter: dict | None = None, n: int = 10) -> list[Document]:
        """Return n random documents, one per course first then fill from any."""
        import random
        all_docs = list(self.vectorstore.docstore._dict.values())
        if metadata_filter:
            clean = {k: v for k, v in metadata_filter.items() if v is not None}
            all_docs = [d for d in all_docs if all(d.metadata.get(k) == v for k, v in clean.items())]
        if not all_docs:
            return []
        # One per course first
        by_course: dict[str, list[Document]] = {}
        for doc in all_docs:
            by_course.setdefault(doc.metadata.get("course_name", ""), []).append(doc)
        courses = list(by_course.keys())
        random.shuffle(courses)
        result = [random.choice(by_course[c]) for c in courses[:n]]
        # Fill remaining from the pool
        if len(result) < n:
            remaining = [d for d in all_docs if d not in result]
            result.extend(random.sample(remaining, min(n - len(result), len(remaining))))
        return result[:n]

    def search(self, query: str, metadata_filter: dict) -> list[Document]:
        """Semantic search with metadata filtering."""
        clean_filter = {k: v for k, v in metadata_filter.items() if v is not None}

        if not query or not query.strip():
            return self._random_docs(clean_filter or None, self.k)

        fetch_k = len(self.vectorstore.docstore._dict)

        # The actual similarity search by the embedding - The meat of the RAG
        if clean_filter:
            results = self.vectorstore.similarity_search(query, k=self.k, filter=clean_filter, fetch_k=fetch_k) # Here we apply the metadata filteration
        else:
            results = self.vectorstore.similarity_search(query, k=self.k)
        return results

    def _format_results(self, docs: list[Document]) -> str:
        """Format search results as a readable string for the LLM."""
        if not docs:
            return "לא נמצאו ביקורות תואמות."
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
            if meta.get("date"):
                header += f" | תאריך: {meta['date'][:10]}"
            parts.append(f"[{header}]\n{doc.page_content}")
        return "\n\n".join(parts)

    def to_langchain_tool(self):
        """Return a LangChain tool the LLM can call to search course reviews."""
        # Capture self for closure
        searcher = self

        @tool
        def search_course_reviews(
            query: str,
            course_name: Optional[str] = None,
            lecturer: Optional[str] = None,
            course_type: Optional[str] = None,
        ) -> str:
            """חפש ביקורות של סטודנטים על קורסים ומרצים.
            השתמש בכלי הזה כדי למצוא מה סטודנטים אומרים על קורס, מרצה, מבחן, תרגילים וכו'.

            Args:
                query: שאילתת חיפוש חופשית - מה המשתמש רוצה לדעת (למשל "מבחן קשה", "מרצה טוב", "קורס קל")
                course_name: שם הקורס לסינון (אופציונלי)
                lecturer: שם המרצה לסינון (אופציונלי)
                course_type: סוג הקורס - חובה/בחירה/סמינר (אופציונלי)
            """
            metadata_filter = {}
            if course_name:
                metadata_filter["course_name"] = course_name
            if lecturer:
                metadata_filter["lecturer"] = lecturer
            if course_type:
                metadata_filter["course_type"] = course_type

            results = searcher.search(query, metadata_filter)
            return searcher._format_results(results)

        # Patch the schema to add enums so the LLM is constrained to valid values
        props = search_course_reviews.args_schema.model_json_schema()["properties"]
        if searcher.all_courses:
            props["course_name"]["enum"] = searcher.all_courses
        if searcher.all_lecturers:
            props["lecturer"]["enum"] = searcher.all_lecturers
        if searcher.all_types:
            props["course_type"]["enum"] = searcher.all_types

        return search_course_reviews
