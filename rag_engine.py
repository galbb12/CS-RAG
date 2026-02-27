"""
Core RAG engine — extracted from main.py so both the CLI and the
FastAPI server can share the same pipeline.
"""

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI

from comment_match_prompt import CommentMatchPrompt
from format_data import parse_documents, parse_grades, parse_kdams
from grades_tool import GradesTool
from kdams_tool import KdamsTool

load_dotenv()

SEMESTER_MAP = {"a": "א׳", "b": "ב׳", "c": "קיץ"}


def _build_kdams_summary(kdams_path: str = None, docs_path: str = None) -> str:
    """Build a compact prerequisite summary for injection into the system prompt."""
    _dir = os.path.dirname(os.path.abspath(__file__))
    if kdams_path is None:
        kdams_path = os.path.join(_dir, "kdams.json")
    if docs_path is None:
        docs_path = os.path.join(_dir, "documents.json")

    with open(kdams_path, encoding="utf-8") as f:
        kdams = json.load(f)

    with open(docs_path, encoding="utf-8") as f:
        raw_docs = json.load(f)
    course_types: dict[str, str] = {}
    for d in raw_docs:
        m = d["metadata"]
        name = m.get("course_name", "")
        ctype = m.get("course_type", "")
        if name and ctype and name not in course_types:
            course_types[name] = ctype

    lines: list[str] = []
    for name, info in kdams.items():
        pts = info.get("credit_points", "?")
        prereqs = info.get("prerequisites") or "אין"
        semesters = info.get("semesters_offered", "")
        sem_str = "+".join(SEMESTER_MAP.get(s, s) for s in semesters) if semesters else "?"
        ctype = course_types.get(name, "")
        type_str = f", {ctype}" if ctype else ""
        note = info.get("note") or ""
        note_str = f" [{note}]" if note else ""
        line = f"- {name} ({pts} נ\"ז, סמסטר {sem_str}{type_str}{note_str}) | קדמים: {prereqs}"
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
2. kdams_tree - חיפוש גרף קדמים של קורס ספציפי (מציג את גרף הקדמים + גרף הקורסים שהוא פותח).
3. course_grades - חיפוש ציונים והתפלגות ציונים של קורס (ממוצע, מספר נבחנים, היסטוגרמה).

תמיד השתמש בכלי לפני שאתה עונה על שאלות לגבי ביקורות או ציונים - אל תנחש תשובות.
לגבי שאלות על קדמים, סדר קורסים, או תכנון לימודים - אתה יכול להשתמש בטבלת הקדמים למטה ישירות.

== תרשימים ==
כשמשתמש שואל על קדמים או מבקש גרף קורסים:
1. תמיד השתמש בכלי kdams_tree קודם כדי לקבל את המידע המדויק
2. בנה את התרשים בפורמט mermaid (הממשק מרנדר אוטומטית כגרף ויזואלי)
3. אל תציין בטקסט שאתה משתמש ב-Mermaid או בכלי - פשוט הצג את התרשים כחלק טבעי מהתשובה

דוגמה (קורסי בסיס למעלה, קורסים מתקדמים למטה):
```mermaid
flowchart TD
  A["מבוא למדעי המחשב"] --> B["מבני נתונים"]
  C["מתמטיקה בדידה"] --> D["אלגוריתמים"]
  B --> D
```

חוקים:
- flowchart TD תמיד
- עטוף שמות בסוגריים מרובעים ומרכאות: ["שם הקורס"]
- כיוון חצים: מקורס בסיס אל הקורס שדורש אותו. כלומר: קדם --> קורס מתקדם (קורסי יסוד למעלה, מתקדמים למטה)
- כל קורס פעם אחת בלבד
- ID קצר באנגלית (A, B, C...) והשם בעברית בתוך הסוגריים
- הוסף תרשים רק כששאלו על קדמים, סדר קורסים, או תכנון לימודים
- אם המשתמש מבקש תרשים של כל הקורסים - עשה זאת בתרשים אחד, הממשק תומך בגלילה וזום

== ציונים ==
כשמשתמש שואל על ציונים, ממוצעים או התפלגות ציונים:
1. השתמש בכלי course_grades כדי לקבל את הנתונים
2. השתמש בפרמטרים לסינון: lecturer, year (שנה עברית כמו תשפד), semester (A/B/C), moed (a/b/c), last_n (מספר מועדים אחרונים)
3. ברירת מחדל: השתמש ב-last_n=5 אלא אם המשתמש ביקש אחרת או ביקש את כל הציונים
4. הכלי מחזיר סיכום טקסטואלי + בלוק histogram שהממשק מרנדר אוטומטית כגרף
5. העתק את בלוק ה-histogram מתוצאת הכלי כמו שהוא לתשובה שלך
6. כדי להציג ציונים בנפרד (למשל לפי שנה או סמסטר), קרא לכלי כמה פעמים עם פילטרים שונים. כך תוכל לכתוב טקסט בין הגרפים.
   דוגמה: קריאה ראשונה עם year=תשפד semester=A, כתיבת ניתוח, קריאה שנייה עם year=תשפד semester=B
7. הוסף ניתוח קצר של הנתונים בעברית (מגמות, השוואות בין מועדים/שנים)

== טבלת קדמים מלאה ==
{kdams_summary}
"""


SUGGESTIONS_PROMPT = """בהתבסס על השיחה, הצע 3 שאלות המשך קצרות שהמשתמש עשוי לשאול.
החזר אך ורק מערך JSON של 3 מחרוזות, ללא טקסט נוסף.
דוגמה: ["שאלה 1?", "שאלה 2?", "שאלה 3?"]"""


@dataclass
class RAGResult:
    """Result of a single RAG invocation."""

    content: str
    tool_outputs: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


class RAGEngine:
    """Stateless RAG engine. Thread-safe for concurrent read-only queries."""

    def __init__(self):
        documents = parse_documents()
        kdams_data = parse_kdams()
        grades_data = parse_grades()

        self.llm = ChatOpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ["LLM_BASE_URL"],
            model=os.environ["LLM_MODEL"],
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.environ["EMBEDDING_API_KEY"],
        )

        comment_matcher = CommentMatchPrompt(embeddings, documents, k=10)
        self.search_tool = comment_matcher.to_langchain_tool()

        kdams = KdamsTool(kdams_data)
        self.kdams_tool = kdams.to_langchain_tool()

        grades = GradesTool(grades_data)
        self.grades_tool = grades.to_langchain_tool()

        self.tools = [self.search_tool, self.kdams_tool, self.grades_tool]
        self.tools_by_name = {t.name: t for t in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            kdams_summary=_build_kdams_summary()
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _openai_to_langchain(self, messages: list[dict]) -> list[BaseMessage]:
        """Convert OpenAI-format messages to LangChain messages."""
        conversation: list[BaseMessage] = [SystemMessage(content=self.system_prompt)]
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                conversation.append(HumanMessage(content=content))
            elif role == "assistant":
                conversation.append(AIMessage(content=content))
            # system messages from the client are ignored — we use our own
        return conversation

    def _run_tool_loop(self, conversation: list[BaseMessage], response):
        """Execute tool calls until the LLM produces a final text answer.

        Returns (final_response, tool_outputs).
        """
        tool_outputs: list[dict] = []
        while response.tool_calls:
            for tool_call in response.tool_calls:
                tool_fn = self.tools_by_name.get(tool_call["name"])
                if tool_fn:
                    result = tool_fn.invoke(tool_call["args"])
                    tool_outputs.append(
                        {
                            "tool_name": tool_call["name"],
                            "tool_args": tool_call["args"],
                            "tool_result": result,
                        }
                    )
                    conversation.append(
                        ToolMessage(content=result, tool_call_id=tool_call["id"])
                    )
            response = self.llm_with_tools.invoke(conversation)
            conversation.append(response)
        return response, tool_outputs

    async def _arun_tool_loop(self, conversation: list[BaseMessage], response):
        """Async version of _run_tool_loop."""
        tool_outputs: list[dict] = []
        while response.tool_calls:
            for tool_call in response.tool_calls:
                tool_fn = self.tools_by_name.get(tool_call["name"])
                if tool_fn:
                    result = tool_fn.invoke(tool_call["args"])
                    tool_outputs.append(
                        {
                            "tool_name": tool_call["name"],
                            "tool_args": tool_call["args"],
                            "tool_result": result,
                        }
                    )
                    conversation.append(
                        ToolMessage(content=result, tool_call_id=tool_call["id"])
                    )
            response = await self.llm_with_tools.ainvoke(conversation)
            conversation.append(response)
        return response, tool_outputs

    def _parse_suggestions(self, text: str) -> list[str]:
        """Extract a JSON array of strings from the LLM response."""
        try:
            start = text.index("[")
            end = text.index("]", start) + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, messages: list[dict]) -> RAGResult:
        """Run a full synchronous RAG query.

        Args:
            messages: OpenAI-format messages [{"role": "...", "content": "..."}]
        """
        conversation = self._openai_to_langchain(messages)
        response = self.llm_with_tools.invoke(conversation)
        conversation.append(response)
        response, tool_outputs = self._run_tool_loop(conversation, response)

        # Generate follow-up suggestions
        suggestions_resp = self.llm.invoke(
            conversation + [HumanMessage(content=SUGGESTIONS_PROMPT)]
        )
        suggestions = self._parse_suggestions(suggestions_resp.content)

        return RAGResult(
            content=response.content,
            tool_outputs=tool_outputs,
            suggestions=suggestions,
        )

    async def aquery(self, messages: list[dict]) -> RAGResult:
        """Run a full async RAG query (non-blocking for the event loop)."""
        conversation = self._openai_to_langchain(messages)
        response = await self.llm_with_tools.ainvoke(conversation)
        conversation.append(response)
        response, tool_outputs = await self._arun_tool_loop(conversation, response)

        # Generate follow-up suggestions
        suggestions_resp = await self.llm.ainvoke(
            conversation + [HumanMessage(content=SUGGESTIONS_PROMPT)]
        )
        suggestions = self._parse_suggestions(suggestions_resp.content)

        return RAGResult(
            content=response.content,
            tool_outputs=tool_outputs,
            suggestions=suggestions,
        )
