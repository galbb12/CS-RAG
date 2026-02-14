import json
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import Field, create_model


class MetadataMatch:
    def __init__(self, documents: list[Document], llm: ChatOpenAI):
        self.documents = documents
        self.lecturers = sorted(set(d.metadata["lecturer"] for d in documents if d.metadata.get("lecturer")))
        self.courses = sorted(set(d.metadata["course_name"] for d in documents if d.metadata.get("course_name")))
        self.course_types = sorted(set(d.metadata["course_type"] for d in documents if d.metadata.get("course_type")))

        filter_schema = self._build_filter_schema()

        self.chain = (
            ChatPromptTemplate.from_messages([
                ("system", "You help search a database of university course reviews. Extract filters from the user's question."),
                ("human", "{query}"),
            ])
            | llm.with_structured_output(filter_schema)
        )

    def _build_filter_schema(self):
        fields = {
            "query": (str, Field(description="The free-text search query (what the user wants to know)")),
        }
        if self.courses:
            fields["course_name"] = (
                Optional[Literal[tuple(self.courses)]],
                Field(default=None, description="The course name, only if the user mentions a specific course"),
            )
        if self.lecturers:
            fields["lecturer"] = (
                Optional[Literal[tuple(self.lecturers)]],
                Field(default=None, description="The lecturer name, only if the user mentions a specific lecturer"),
            )
        if self.course_types:
            fields["course_type"] = (
                Optional[Literal[tuple(self.course_types)]],
                Field(default=None, description="The course type, only if the user specifies"),
            )
        return create_model("CourseReviewFilters", **fields)

    def _filter_documents(self, filters) -> list[Document]:
        results = self.documents
        if getattr(filters, "course_name", None):
            results = [d for d in results if d.metadata.get("course_name") == filters.course_name]
        if getattr(filters, "lecturer", None):
            results = [d for d in results if d.metadata.get("lecturer") == filters.lecturer]
        if getattr(filters, "course_type", None):
            results = [d for d in results if d.metadata.get("course_type") == filters.course_type]
        return results

    def extract_filters(self, prompt: str):
        """Return the raw pydantic filter object extracted by the LLM."""
        filters = self.chain.invoke({"query": prompt})
        print(f"Found the filters: {filters}")
        return filters

    def run(self, prompt: str) -> list[Document]:
        filters = self.extract_filters(prompt)
        return self._filter_documents(filters)


# ---- Demo ----
if __name__ == "__main__":
    load_dotenv()

    with open("documents.json", encoding="utf-8") as f:
        raw_docs = json.load(f)

    documents = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in raw_docs]

    llm = ChatOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-2.5-flash",
    )

    metadata_match = MetadataMatch(documents, llm)

    examples = [
        "מה אומרים על אור דונקלמן?",
        "קורסי בחירה קלים",
        "איך המבחן במערכות הפעלה?",
        "האם שולי וינטנר מרצה טוב?",
    ]

    for question in examples:
        print(f"\n{'='*60}")
        print(f"Question: {question}")

        filtered = metadata_match.run(question)
        print(f"Matched {len(filtered)} documents (from {len(documents)} total)")

        if filtered:
            print(f"First match preview: {filtered[0].page_content}")
