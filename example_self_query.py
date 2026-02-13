"""
Example: LLM extracts structured filters from user question using function calling.
The enums force the LLM to only pick valid values from our data.
"""

import json
from openai import OpenAI
from dotenv import load_dotenv


# ---- Load enum values from our data ----
with open("documents.json", encoding="utf-8") as f:
    docs = json.load(f)

ALL_LECTURERS = sorted(set(d["metadata"]["lecturer"] for d in docs if d["metadata"]["lecturer"]))
ALL_COURSES = sorted(set(d["metadata"]["course_name"] for d in docs if d["metadata"]["course_name"]))
ALL_TYPES = sorted(set(d["metadata"]["course_type"] for d in docs if d["metadata"]["course_type"]))

# ---- Define the tool with enums ----
search_tool = {
    "type": "function",
    "function": {
        "name": "search_course_reviews",
        "description": (
            "Search for course reviews. Extract any filters the user mentions. "
            "Only set a filter if the user clearly refers to it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The free-text search query (what the user wants to know)",
                },
                "course_name": {
                    "type": "string",
                    "description": "The course name, only if the user mentions a specific course",
                    "enum": ALL_COURSES,
                },
                "lecturer": {
                    "type": "string",
                    "description": "The lecturer name, only if the user mentions a specific lecturer",
                    "enum": ALL_LECTURERS,
                },
                "course_type": {
                    "type": "string",
                    "description": "The course type, only if the user specifies",
                    "enum": ALL_TYPES,
                },
            },
            "required": ["query"],
        },
    },
}


def extract_filters(user_question: str) -> dict:
    """Send the user's question to the LLM and get back structured filters."""
    client = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {
                "role": "system",
                "content": "You help search a database of university course reviews. Extract filters from the user's question.",
            },
            {"role": "user", "content": user_question},
        ],
        tools=[search_tool],
        tool_choice={"type": "function", "function": {"name": "search_course_reviews"}},
    )

    tool_call = response.choices[0].message.tool_calls[0]
    return json.loads(tool_call.function.arguments)


def filter_documents(docs: list[dict], filters: dict) -> list[dict]:
    """Filter documents by the extracted metadata."""
    results = docs
    if "course_name" in filters:
        results = [d for d in results if d["metadata"]["course_name"] == filters["course_name"]]
    if "lecturer" in filters:
        results = [d for d in results if d["metadata"]["lecturer"] == filters["lecturer"]]
    if "course_type" in filters:
        results = [d for d in results if d["metadata"]["course_type"] == filters["course_type"]]
    return results


# ---- Demo ----
if __name__ == "__main__":
    load_dotenv()

    examples = [
        "מה אומרים על אור דונקלמן?",
        "קורסי בחירה קלים",
        "איך המבחן במערכות הפעלה?",
        "האם שולי וינטנר מרצה טוב?",
    ]

    for question in examples:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        filters = extract_filters(question)
        print(f"LLM extracted: {json.dumps(filters, ensure_ascii=False, indent=2)}")

        filtered = filter_documents(docs, filters)
        print(f"Matched {len(filtered)} documents (from {len(docs)} total)")

        if filtered:
            print(f"First match preview: {filtered[0]['page_content']}")
