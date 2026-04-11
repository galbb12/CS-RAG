import hashlib
import os
import time
import threading
from typing import Optional

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS


FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index"))


class CommentMatchPrompt:
    """Pre-embeds all documents once. Exposes a LangChain tool for the LLM to search."""

    def __init__(self, embeddings: Embeddings, documents: list[Document], k: int = 10):
        self.embeddings = embeddings
        self.k = k
        self._lock = threading.RLock()
        self._tool_props = None
        self.vectorstore, self.documents = self._load_or_build_index(documents)

        self.all_lecturers = sorted(set(d.metadata["lecturer"] for d in self.documents if d.metadata.get("lecturer")))
        self.all_courses = sorted(set(d.metadata["course_name"] for d in self.documents if d.metadata.get("course_name")))
        self.all_types = sorted(set(d.metadata["course_type"] for d in self.documents if d.metadata.get("course_type")))

    def _embed_batches(self, documents: list[Document], into: FAISS | None = None) -> FAISS:
        """Embed documents in batches with rate-limit retry. Adds into existing index if provided."""
        batch_size = 80
        store = into
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            print(f"  Embedding batch {i // batch_size + 1} ({len(batch)} docs)...")
            while True:
                try:
                    if store is None:
                        store = FAISS.from_documents(batch, self.embeddings)
                    else:
                        store.add_documents(batch)
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        print("  Rate limited, waiting 60s...")
                        time.sleep(60)
                    else:
                        raise
            if i + batch_size < len(documents):
                time.sleep(61)
        assert store is not None
        return store

    def _save(self, store: FAISS):
        """Save FAISS index. Strips content before saving so pkl has no raw data."""
        from langchain_community.docstore.in_memory import InMemoryDocstore

        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

        # Build a stripped copy of the docstore (no content, with hash for matching)
        stripped = {}
        for doc_id, doc in store.docstore._dict.items():
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            meta = {**doc.metadata, "date": str(doc.metadata.get("date", "")), "_hash": content_hash}
            stripped[doc_id] = Document(page_content="", metadata=meta)

        # Create a temporary FAISS store with stripped docstore and save it
        temp = FAISS(
            embedding_function=store.embedding_function,
            index=store.index,
            docstore=InMemoryDocstore(stripped),
            index_to_docstore_id=dict(store.index_to_docstore_id),
        )
        temp.save_local(FAISS_INDEX_PATH)

    def _load_or_build_index(self, documents: list[Document]) -> tuple[FAISS, list[Document]]:
        """Load saved index or build from scratch."""
        try:
            store = FAISS.load_local(FAISS_INDEX_PATH, self.embeddings, allow_dangerous_deserialization=True)
            print(f"Loaded FAISS index ({store.index.ntotal} vectors)")

            # Restore real content by matching content hash from pkl to SQL documents
            hash_to_doc = {hashlib.md5(d.page_content.encode()).hexdigest(): d for d in documents}
            for doc_id, doc in store.docstore._dict.items():
                h = doc.metadata.get("_hash")
                if h and h in hash_to_doc:
                    store.docstore._dict[doc_id] = hash_to_doc[h]

            return store, documents

        except Exception:
            print(f"Building FAISS index for {len(documents)} documents...")
            store = self._embed_batches(documents)
            self._save(store)
            return store, documents

    def add_new_documents(self, new_docs: list[Document]):
        """Incrementally embed and add new documents. Thread-safe."""
        if not new_docs:
            return
        print(f"Adding {len(new_docs)} new documents to index...")
        with self._lock:
            self._embed_batches(new_docs, into=self.vectorstore)
            self.documents = self.documents + new_docs
            self._save(self.vectorstore)

            self.all_lecturers = sorted(set(d.metadata["lecturer"] for d in self.documents if d.metadata.get("lecturer")))
            self.all_courses = sorted(set(d.metadata["course_name"] for d in self.documents if d.metadata.get("course_name")))
            self.all_types = sorted(set(d.metadata["course_type"] for d in self.documents if d.metadata.get("course_type")))
            self._update_enums()
        print(f"Index updated ({self.vectorstore.index.ntotal} vectors total)")

    def _indexed_hashes(self) -> set[str]:
        """Get content hashes of all documents currently in the index."""
        return {
            hashlib.md5(doc.page_content.encode()).hexdigest()
            for doc in self.vectorstore.docstore._dict.values()
            if doc.page_content
        }

    def find_new_documents(self, all_docs: list[Document]) -> list[Document]:
        """Compare documents against indexed hashes, return only unindexed ones."""
        with self._lock:
            known = self._indexed_hashes()
        return [d for d in all_docs if hashlib.md5(d.page_content.encode()).hexdigest() not in known]

    def _random_docs(self, metadata_filter: dict | None = None, n: int = 10) -> list[Document]:
        """Return n random documents, one per course first then fill from any."""
        import random
        with self._lock:
            all_docs = list(self.vectorstore.docstore._dict.values())
        if metadata_filter:
            clean = {k: v for k, v in metadata_filter.items() if v is not None}
            all_docs = [d for d in all_docs if all(d.metadata.get(k) == v for k, v in clean.items())]
        if not all_docs:
            return []
        by_course: dict[str, list[Document]] = {}
        for doc in all_docs:
            by_course.setdefault(doc.metadata.get("course_name", ""), []).append(doc)
        courses = list(by_course.keys())
        random.shuffle(courses)
        result = [random.choice(by_course[c]) for c in courses[:n]]
        if len(result) < n:
            remaining = [d for d in all_docs if d not in result]
            result.extend(random.sample(remaining, min(n - len(result), len(remaining))))
        return result[:n]

    def search(self, query: str, metadata_filter: dict) -> list[Document]:
        """Semantic search with metadata filtering."""
        clean_filter = {k: v for k, v in metadata_filter.items() if v is not None}

        if not query or not query.strip():
            return self._random_docs(clean_filter or None, self.k)

        with self._lock:
            fetch_k = len(self.vectorstore.docstore._dict)
            if clean_filter:
                results = self.vectorstore.similarity_search(query, k=self.k, filter=clean_filter, fetch_k=fetch_k)
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
            if meta.get("author"):
                header += f" | כותב: {meta['author']}"
            if meta.get("date"):
                header += f" | תאריך: {str(meta['date'])[:10]}"
            parts.append(f"[{header}]\n{doc.page_content}")
        return "\n\n".join(parts)

    def to_langchain_tool(self):
        """Return a LangChain tool the LLM can call to search course reviews."""
        searcher = self

        @tool
        def search_course_reviews(
            query: str,
            course_name: Optional[str] = None,
            lecturer: Optional[str] = None,
            course_type: Optional[str] = None,
            written_by: Optional[str] = None,
        ) -> str:
            """חפש ביקורות של סטודנטים על קורסים ומרצים.
            השתמש בכלי הזה כדי למצוא מה סטודנטים אומרים על קורס, מרצה, מבחן, תרגילים וכו'.

            Args:
                query: שאילתת חיפוש חופשית - מה המשתמש רוצה לדעת (למשל "מבחן קשה", "מרצה טוב", "קורס קל")
                course_name: שם הקורס לסינון (אופציונלי)
                lecturer: שם המרצה לסינון (אופציונלי)
                course_type: סוג הקורס - חובה/בחירה/סמינר (אופציונלי)
                written_by: שם כותב הביקורת לסינון (אופציונלי)
            """
            metadata_filter = {}
            if course_name:
                metadata_filter["course_name"] = course_name
            if lecturer:
                metadata_filter["lecturer"] = lecturer
            if course_type:
                metadata_filter["course_type"] = course_type
            results = searcher.search(query, metadata_filter)

            if written_by:
                results = [d for d in results if written_by in d.metadata.get("author", "")]

            return searcher._format_results(results)

        self._tool_props = search_course_reviews.args_schema.model_json_schema()["properties"]
        self._update_enums()

        return search_course_reviews

    def _update_enums(self):
        """Update tool schema enums to reflect current data."""
        if self._tool_props:
            self._tool_props["course_name"]["enum"] = self.all_courses
            self._tool_props["lecturer"]["enum"] = self.all_lecturers
            self._tool_props["course_type"]["enum"] = self.all_types
