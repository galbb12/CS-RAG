import json
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI

from comment_match_prompt import CommentMatchPrompt
from kdams_tool import KdamsTool

load_dotenv()

SEMESTER_MAP = {"a": "א׳", "b": "ב׳", "c": "קיץ"}

def _build_kdams_summary(kdams_path: str = "kdams.json", docs_path: str = "documents.json") -> str:
    """Build a compact prerequisite summary for injection into the system prompt."""
    with open(kdams_path, encoding="utf-8") as f:
        kdams = json.load(f)

    # Extract course_type from documents metadata
    with open(docs_path, encoding="utf-8") as f:
        raw_docs = json.load(f)
    course_types = {}
    for d in raw_docs:
        m = d["metadata"]
        name = m.get("course_name", "")
        ctype = m.get("course_type", "")
        if name and ctype and name not in course_types:
            course_types[name] = ctype

    lines = []
    for name, info in kdams.items():
        pts = info.get("credit_points", "?")
        prereqs = info.get("prerequisites") or "אין"
        semesters = info.get("semesters_offered", "")
        sem_str = "+".join(SEMESTER_MAP.get(s, s) for s in semesters) if semesters else "?"
        ctype = course_types.get(name, "")
        type_str = f", {ctype}" if ctype else ""
        line = f"- {name} ({pts} נ\"ז, סמסטר {sem_str}{type_str}) | קדמים: {prereqs}"
        lines.append(line)
    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """אתה עוזר לסטודנטים בחוג למדעי המחשב באוניברסיטת חיפה.
אתה עונה על שאלות לגבי קורסים, מרצים, מבחנים ועוד על סמך ביקורות אמיתיות של סטודנטים.

כללים:
- ענה רק על סמך מידע שקיבלת מהכלים או מעץ הקדמים למטה. אל תמציא מידע.
- אם אין מספיק מידע, אמור זאת בכנות.
- ציין מגמות חוזרות בביקורות (למשל אם כמה סטודנטים מסכימים על נקודה מסוימת).
- השתמש בעברית בתשובות שלך.
- היה תמציתי ועניני.

הכלים שלך:
1. search_course_reviews - חיפוש ביקורות סטודנטים. אתה יכול לסנן לפי שם קורס, מרצה, או סוג קורס.
2. kdams_tree - חיפוש עץ קדמים של קורס ספציפי (מציג עץ מלא + קורסים שהוא פותח).

תמיד השתמש בכלי לפני שאתה עונה על שאלות לגבי ביקורות - אל תנחש תשובות.
לגבי שאלות על קדמים, סדר קורסים, או תכנון לימודים - אתה יכול להשתמש בטבלת הקדמים למטה ישירות.

== טבלת קדמים מלאה ==
{kdams_summary}
"""


def main():
    # Load documents
    with open("documents.json", encoding="utf-8") as f:
        raw_docs = json.load(f)
    documents = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in raw_docs]

    # Init LLM + embeddings
    llm = ChatOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-2.5-flash",
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.environ["OPENAI_API_KEY"],
    )

    # Build tools
    comment_matcher = CommentMatchPrompt(embeddings, documents, k=10)
    search_tool = comment_matcher.to_langchain_tool()

    kdams = KdamsTool()
    kdams_tool = kdams.to_langchain_tool()

    tools = [search_tool, kdams_tool]
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    kdams_summary = _build_kdams_summary()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(kdams_summary=kdams_summary)
    conversation = [SystemMessage(content=system_prompt)]

    print("CS-RAG Bot Ready! (type 'exit' to quit)\n")

    while True:
        question = input("שאלה: ").strip()
        if not question or question == "exit":
            break

        conversation.append(HumanMessage(content=question))

        # Let the LLM respond (may call tools)
        response = llm_with_tools.invoke(conversation)
        conversation.append(response)

        # Handle tool calls in a loop
        while response.tool_calls:
            for tool_call in response.tool_calls:
                tool_fn = tools_by_name.get(tool_call["name"])
                if tool_fn:
                    result = tool_fn.invoke(tool_call["args"])
                    conversation.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

            response = llm_with_tools.invoke(conversation)
            conversation.append(response)

        print(f"\n{response.content}\n")


if __name__ == "__main__":
    main()
