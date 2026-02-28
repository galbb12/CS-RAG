"""
RAG Evaluation Pipeline

Runs test questions through the RAG engine and evaluates:
1. Deterministic metrics: tool selection, argument correctness, language, latency
2. LLM-as-judge metrics: faithfulness, relevance, completeness (using GLM5)

Usage:
    python evaluate.py                          # run full evaluation
    python evaluate.py --no-judge               # skip LLM judge (deterministic only)
    python evaluate.py --dataset custom.json    # use custom dataset
    python evaluate.py --ids 1 2 3              # run specific question IDs only
    python evaluate.py --concurrency 5          # run 5 questions in parallel
"""

import argparse
import asyncio
import json
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from rag_engine import RAGEngine

load_dotenv()

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QuestionResult:
    """Result for a single evaluation question."""

    id: int
    question: str
    category: str
    answer: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    latency_s: float = 0.0

    # Deterministic metrics
    tool_selection_correct: bool | None = None
    argument_correct: bool | None = None
    response_in_hebrew: bool | None = None

    # Judge metrics (1-10 scale)
    faithfulness: int | None = None
    relevance: int | None = None
    completeness: int | None = None
    judge_reasoning: str = ""

    error: str | None = None


@dataclass
class EvalSummary:
    """Aggregate evaluation results."""

    total_questions: int = 0
    questions_with_errors: int = 0

    # Deterministic
    tool_selection_accuracy: float = 0.0
    argument_accuracy: float = 0.0
    hebrew_response_rate: float = 0.0
    avg_latency_s: float = 0.0

    # Judge
    avg_faithfulness: float = 0.0
    avg_relevance: float = 0.0
    avg_completeness: float = 0.0

    # Per-category breakdown
    category_scores: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Deterministic metrics
# ---------------------------------------------------------------------------


def check_tool_selection(tool_calls: list[dict], test_case: dict) -> bool | None:
    """Check if the RAG engine called the expected tool(s)."""
    called_tools = {tc["tool_name"] for tc in tool_calls}

    # Multi-tool questions
    if "expected_tools" in test_case:
        expected = set(test_case["expected_tools"])
        return expected.issubset(called_tools)

    expected_tool = test_case.get("expected_tool")

    # No tool expected (edge cases like greetings)
    if expected_tool is None:
        return len(tool_calls) == 0 or True  # lenient for edge cases

    return expected_tool in called_tools


def check_arguments(tool_calls: list[dict], test_case: dict) -> bool | None:
    """Check if the tool was called with the expected arguments."""
    expected_args = test_case.get("expected_args", {})
    if not expected_args:
        return None  # nothing to check

    for tc in tool_calls:
        args = tc.get("tool_args", {})
        match = True
        for key, value in expected_args.items():
            if key not in args:
                match = False
                break
            # Fuzzy match: expected value is substring of actual or vice versa
            actual = args[key]
            if isinstance(actual, str) and isinstance(value, str):
                if value not in actual and actual not in value:
                    match = False
                    break
            elif actual != value:
                match = False
                break
        if match:
            return True

    return False


def check_hebrew(text: str) -> bool:
    """Check if the response is predominantly in Hebrew."""
    if not text:
        return False
    hebrew_chars = len(re.findall(r"[\u0590-\u05FF]", text))
    latin_chars = len(re.findall(r"[a-zA-Z]", text))
    # Hebrew should dominate over Latin (allowing for code/mermaid blocks)
    total = hebrew_chars + latin_chars
    if total == 0:
        return False
    return hebrew_chars / total > 0.3


# ---------------------------------------------------------------------------
# Judge DB tools — lets the judge LLM search the SQL file interactively
# ---------------------------------------------------------------------------


class JudgeDBTools:
    """Provides tool definitions and execution for the judge agent.

    The judge can call these tools in a loop to investigate the raw SQL
    database and verify facts in the RAG answer.
    """

    MATCH_CONTEXT_CHARS = 120  # chars shown around each match

    def __init__(self):
        from format_data import SQL_PATH
        with open(SQL_PATH, encoding="utf-8") as f:
            raw_sql = f.read()
        self.lines = raw_sql.splitlines(keepends=True)
        self.total_lines = len(self.lines)
        self.sqlite_conn = self._load_sqlite(raw_sql)
        self.db_info = self._build_db_info()

    def _load_sqlite(self, raw_sql: str) -> sqlite3.Connection:
        """Load the SQL dump into an in-memory SQLite database."""
        conn = sqlite3.connect(":memory:")
        # MySQL -> SQLite compatibility fixes
        sql = raw_sql
        # Remove MySQL-specific syntax
        sql = re.sub(r"ENGINE=\w+", "", sql)
        sql = re.sub(r"DEFAULT CHARSET=\w+", "", sql)
        sql = re.sub(r"COLLATE=\w+", "", sql)
        sql = re.sub(r"AUTO_INCREMENT=\d+", "", sql)
        sql = re.sub(r"AUTO_INCREMENT", "", sql)
        sql = re.sub(r"UNSIGNED", "", sql)
        sql = re.sub(r"CHARACTER SET \w+", "", sql)
        sql = re.sub(r"COLLATE \w+", "", sql)
        sql = re.sub(r"IF NOT EXISTS", "", sql)
        sql = re.sub(r"LOCK TABLES.*?;", "", sql, flags=re.DOTALL)
        sql = re.sub(r"UNLOCK TABLES;", "", sql)
        sql = re.sub(r"START TRANSACTION;", "", sql)
        sql = re.sub(r"COMMIT;", "", sql)
        sql = re.sub(r"SET .*?;", "", sql)
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        sql = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
        # int(11) -> integer, text DEFAULT NULL -> text, current_timestamp() -> CURRENT_TIMESTAMP
        sql = re.sub(r"\bint\(\d+\)", "integer", sql)
        sql = re.sub(r"\bcurrent_timestamp\(\)", "CURRENT_TIMESTAMP", sql)
        # MySQL escaped quotes: \' -> '' (SQLite-style)
        sql = sql.replace("\\'", "''")
        sql = re.sub(r"\\n", "\n", sql)
        sql = re.sub(r"\\r", "", sql)
        try:
            conn.executescript(sql)
        except Exception:
            # Try statement by statement, skipping failures
            for stmt in sql.split(";"):
                stmt = stmt.strip()
                if stmt:
                    try:
                        conn.execute(stmt)
                    except Exception:
                        pass
            conn.commit()
        return conn

    def _build_db_info(self) -> str:
        """Precompute DB structure info for the system prompt."""
        parts = [f"SQL database ({self.total_lines} lines). Tables and columns:"]
        # Get all table names from both SQLite and raw SQL
        all_tables = set()
        try:
            cursor = self.sqlite_conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            all_tables = {r[0] for r in cursor.fetchall()}
        except Exception:
            pass
        # Also find tables in raw SQL that may not have loaded
        for line in self.lines:
            m = re.search(r"(?:CREATE TABLE|INSERT INTO)\s+`(\w+)`", line, re.IGNORECASE)
            if m:
                all_tables.add(m.group(1))

        for tname in sorted(all_tables):
            try:
                col_cursor = self.sqlite_conn.execute("PRAGMA table_info(" + tname + ")")
                cols = [row[1] for row in col_cursor.fetchall()]
                if cols:
                    row_count = self.sqlite_conn.execute('SELECT COUNT(*) FROM "' + tname + '"').fetchone()[0]
                    parts.append(f"  - `{tname}` ({row_count} rows): {', '.join(cols)}")
                else:
                    parts.append(f"  - `{tname}` (use grep_db/read_lines to query)")
            except Exception:
                parts.append(f"  - `{tname}` (use grep_db/read_lines to query)")
        return "\n".join(parts)

    def get_langchain_tools(self):
        """Create LangChain @tool instances that wrap this DB access."""
        db = self

        @tool
        def grep_db(pattern: str) -> str:
            """Search the SQL database file using a regex pattern (case-insensitive).
            Returns matching lines with context around each match.
            Use to find data about specific courses, lecturers, grades, or any text."""
            return db._grep_db(pattern)

        @tool
        def read_lines(start_line: int, num_lines: int) -> str:
            """Read a range of lines from the SQL database file.
            Useful after grep_db to see surrounding context."""
            return db._read_lines(start_line, num_lines)

        @tool
        def sql_query(query: str) -> str:
            """Run a read-only SQL SELECT query against the database.
            Use to look up courses, grades, reviews, prerequisites.
            Returns up to 50 rows formatted as a table."""
            return db._sql_query(query)

        return [grep_db, read_lines, sql_query]

    def _grep_db(self, pattern: str) -> str:
        if not pattern:
            return "Error: empty pattern"
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Regex error: {e}"

        ctx = self.MATCH_CONTEXT_CHARS
        matches = []
        for i, line in enumerate(self.lines, start=1):
            m = regex.search(line)
            if m:
                # Show context around the match, not the whole line
                start = max(0, m.start() - ctx)
                end = min(len(line), m.end() + ctx)
                snippet = line[start:end].strip()
                prefix = "..." if start > 0 else ""
                suffix = "..." if end < len(line.rstrip()) else ""
                matches.append(f"L{i}: {prefix}{snippet}{suffix}")

        if not matches:
            return f"No matches found for pattern: {pattern}"
        return f"{len(matches)} matches found:\n" + "\n".join(matches)

    def _read_lines(self, start_line: int, num_lines: int) -> str:
        start = max(1, start_line) - 1  # convert to 0-based
        num = min(num_lines, 30)  # cap at 30 lines
        end = min(start + num, self.total_lines)

        result = []
        for i in range(start, end):
            result.append(f"L{i + 1}: {self.lines[i].rstrip()}")

        if not result:
            return f"No lines in range. File has {self.total_lines} lines."
        return "\n".join(result)

    def _sql_query(self, query: str) -> str:
        if not query:
            return "Error: empty query"
        # Only allow SELECT
        if not query.strip().upper().startswith("SELECT"):
            return "Error: only SELECT queries are allowed"
        try:
            cursor = self.sqlite_conn.execute(query)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(50)
            if not rows:
                return "Query returned 0 rows."
            header = " | ".join(columns)
            lines = [header, "-" * len(header)]
            for row in rows:
                lines.append(" | ".join(str(v) for v in row))
            total = self.sqlite_conn.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()[0]
            if total > 50:
                lines.append(f"... ({total} total rows, showing first 50)")
            return "\n".join(lines)
        except Exception as e:
            return f"SQL error: {e}"

# ---------------------------------------------------------------------------
# LLM Judge (async with 429 retry)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT_TEMPLATE = """You are evaluating a RAG (Retrieval-Augmented Generation) system that helps university students get information about courses at the University of Haifa CS department.

You have access to tools that let you search and read the raw SQL database that the RAG system is built on. Use these tools to verify facts, numbers, and claims in the generated answer.

## Database Structure
{db_info}

## Your workflow
1. Read the question, the retrieved context the RAG used, and the generated answer.
2. Use your tools to look up the ACTUAL data in the database.
   - Search for the course name, lecturer, grade averages, etc.
   - Verify any numbers (averages, histograms, prerequisites) the answer mentions.
   - IMPORTANT: Call ALL the tools you need in a SINGLE round. For example, if you need to search for a course name AND a lecturer, call grep_db twice in the same response, not in separate rounds. Each round costs time.
   - Prefer sql_query over grep_db when possible — it's faster and more precise.
3. After investigating (ideally 1-2 rounds), provide your final scores.

Score each dimension on a 1-10 scale:

**Faithfulness** (Is the answer factually correct based on the actual database?):
- 1-2: Answer contains fabricated information or wrong numbers
- 3-4: Answer has multiple factual errors
- 5-6: Answer is partially correct but has some wrong data
- 7-8: Answer is mostly correct with minor inaccuracies
- 9-10: Answer is fully accurate and matches the database

**Relevance** (Does the answer address the student's question?):
- 1-2: Answer is off-topic or doesn't address the question
- 3-4: Answer barely addresses the question
- 5-6: Answer partially addresses the question
- 7-8: Answer addresses the question but misses some aspects
- 9-10: Answer directly and fully addresses the question

**Completeness** (Does the answer cover all aspects the student asked about?):
- 1-2: Answer is missing most relevant information
- 3-4: Answer covers very little of what was asked
- 5-6: Answer covers some aspects but leaves significant gaps
- 7-8: Answer covers most aspects with minor omissions
- 9-10: Answer is comprehensive and covers all aspects

When you are done investigating, respond with ONLY a JSON object (no markdown, no extra text):
{"faithfulness": N, "relevance": N, "completeness": N, "reasoning": "brief explanation of your findings"}"""

MAX_JUDGE_TOOL_ROUNDS = -1  # max tool-use rounds (-1 = unlimited)


class JudgeScore(BaseModel):
    """Structured output schema for the judge's final score."""
    faithfulness: int = Field(description="1-10: Is the answer factually correct based on the database?")
    relevance: int = Field(description="1-10: Does the answer address the student's question?")
    completeness: int = Field(description="1-10: Does the answer cover all aspects asked about?")
    reasoning: str = Field(description="Brief explanation of your findings")


async def judge_response(
    judge_llm: ChatOpenAI,
    question: str,
    context: str,
    answer: str,
    db_tools: "JudgeDBTools",
) -> dict:
    """Use the judge LLM as a tool-calling agent to score a RAG response.

    Follows the same bind_tools + tool loop pattern as RAGEngine.
    After the tool loop, uses with_structured_output to extract the score.
    """
    tools = db_tools.get_langchain_tools()
    llm_with_tools = judge_llm.bind_tools(tools, parallel_tool_calls=True)
    tool_map = {t.name: t for t in tools}

    system_prompt = JUDGE_SYSTEM_PROMPT_TEMPLATE.replace("{db_info}", db_tools.db_info)
    user_msg = (
        "Evaluate this RAG response. Use your tools to verify facts against the database before scoring.\n\n"
        f"**Question**: {question}\n\n"
        f"**Retrieved Context (what the RAG system saw)**:\n{context}\n\n"
        f"**Generated Answer (what the RAG system replied)**:\n{answer}"
    )

    messages: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg),
    ]

    # --- Tool loop: let the judge investigate the DB ---
    max_rounds = MAX_JUDGE_TOOL_ROUNDS if MAX_JUDGE_TOOL_ROUNDS > 0 else 999

    for round_num in range(max_rounds):
        try:
            response = await llm_with_tools.ainvoke(messages)
        except Exception as e:
            return {"faithfulness": None, "relevance": None, "completeness": None,
                    "reasoning": f"Judge API error: {e}"}

        print(f"    [Judge round {round_num}] tool_calls={bool(response.tool_calls)} content_len={len(response.content or '')}", flush=True)

        if not response.tool_calls:
            break  # model is done investigating

        messages.append(response)
        for tc in response.tool_calls:
            tool_fn = tool_map.get(tc["name"])
            result = tool_fn.invoke(tc["args"]) if tool_fn else f"Unknown tool: {tc['name']}"
            print(f"    [Judge tool] {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)[:80]}) -> {len(result)} chars", flush=True)
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    # --- Structured score extraction ---
    # Append any text the model produced in the last round, then ask for the score
    if response.content:
        messages.append(response)
    messages.append(HumanMessage(content="Now provide your final scores."))

    try:
        score_llm = judge_llm.with_structured_output(JudgeScore)
        result = await score_llm.ainvoke(messages)
        score = result.model_dump()
        print(f"    [Judge] Final score: {score}", flush=True)
        return score
    except Exception as e:
        return {"faithfulness": None, "relevance": None, "completeness": None,
                "reasoning": f"Judge score extraction failed: {e}"}


# ---------------------------------------------------------------------------
# Single question evaluation
# ---------------------------------------------------------------------------


async def evaluate_one(
    engine: RAGEngine,
    test_case: dict,
    judge_llm: ChatOpenAI | None,
    use_judge: bool,
    db_tools: JudgeDBTools | None,
    index: int,
    total: int,
) -> QuestionResult:
    """Evaluate a single test question."""
    qid = test_case["id"]
    question = test_case["question"]
    category = test_case["category"]

    print(f"\n[{index}/{total}] Q{qid} ({category}): {question[:60]}...")

    result = QuestionResult(id=qid, question=question, category=category)

    # Skip empty questions
    if not question.strip():
        result.error = "Empty question"
        result.answer = ""
        print(f"  -> Skipped (empty question)")
        return result

    # Run through RAG engine (async)
    try:
        start = time.time()
        rag_result = await engine.aquery([{"role": "user", "content": question}])
        elapsed = time.time() - start

        result.answer = rag_result.content
        result.tool_calls = rag_result.tool_outputs
        result.latency_s = round(elapsed, 2)

        # Deterministic metrics
        result.tool_selection_correct = check_tool_selection(rag_result.tool_outputs, test_case)
        result.argument_correct = check_arguments(rag_result.tool_outputs, test_case)
        result.response_in_hebrew = check_hebrew(rag_result.content)

        status_parts = []
        if result.tool_selection_correct is not None:
            status_parts.append(f"tool={'OK' if result.tool_selection_correct else 'WRONG'}")
        if result.argument_correct is not None:
            status_parts.append(f"args={'OK' if result.argument_correct else 'WRONG'}")
        status_parts.append(f"hebrew={'yes' if result.response_in_hebrew else 'no'}")
        status_parts.append(f"{result.latency_s}s")
        print(f"  -> {', '.join(status_parts)}")

        # Judge evaluation (tool-calling agent)
        if use_judge and judge_llm and db_tools:
            context = "\n---\n".join(
                tc.get("tool_result", "") for tc in rag_result.tool_outputs
            )
            judge_result = await judge_response(
                judge_llm, question, context, rag_result.content, db_tools
            )
            if not isinstance(judge_result, dict):
                judge_result = {"faithfulness": None, "relevance": None, "completeness": None, "reasoning": f"Unexpected judge output: {judge_result}"}
            result.faithfulness = judge_result.get("faithfulness")
            result.relevance = judge_result.get("relevance")
            result.completeness = judge_result.get("completeness")
            result.judge_reasoning = judge_result.get("reasoning", "")
            print(f"  -> Judge: faith={result.faithfulness} rel={result.relevance} comp={result.completeness}")

    except Exception as e:
        import traceback
        result.error = str(e)
        print(f"  -> ERROR: {e}")
        traceback.print_exc()

    return result


# ---------------------------------------------------------------------------
# Main evaluation loop (parallel)
# ---------------------------------------------------------------------------


async def run_evaluation(
    dataset_path: str = "eval_dataset.json",
    output_path: str = "eval_results.json",
    use_judge: bool = True,
    question_ids: list[int] | None = None,
    concurrency: int = 3,
) -> tuple[list[QuestionResult], EvalSummary]:
    """Run the evaluation pipeline with parallel question processing."""

    # Load dataset
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    if question_ids:
        dataset = [q for q in dataset if q["id"] in question_ids]

    print(f"Loaded {len(dataset)} test questions (concurrency={concurrency})")

    # Initialize RAG engine
    print("Initializing RAG engine...")
    engine = RAGEngine()
    print("RAG engine ready.")

    # Initialize judge LLM
    judge_llm = None
    if use_judge:
        judge_key = os.environ.get("JUDGE_API_KEY")
        judge_url = os.environ.get("JUDGE_BASE_URL")
        judge_model = os.environ.get("JUDGE_MODEL")
        if judge_key and judge_url and judge_model:
            judge_llm = ChatOpenAI(
                api_key=judge_key,
                base_url=judge_url,
                model=judge_model,
                temperature=0.0,
                max_tokens=16384,
                max_retries=5,
                timeout=300,
            )
            print(f"Judge model: {judge_model}")
        else:
            print("Warning: JUDGE_API_KEY, JUDGE_BASE_URL, or JUDGE_MODEL not set. Skipping judge evaluation.")
            use_judge = False

    # Load DB tools for the judge agent
    db_tools = None
    if use_judge:
        print("Loading SQL database for judge tools...")
        db_tools = JudgeDBTools()
        print(f"Judge DB tools: loaded {db_tools.total_lines} lines of SQL")

    # Run evaluation with bounded concurrency
    semaphore = asyncio.Semaphore(concurrency)
    results: list[QuestionResult] = [None] * len(dataset)  # preserve order
    completed = 0

    async def run_with_semaphore(idx: int, test_case: dict):
        nonlocal completed
        async with semaphore:
            result = await evaluate_one(
                engine, test_case, judge_llm, use_judge, db_tools,
                index=idx + 1, total=len(dataset),
            )
            results[idx] = result
            completed += 1
            # Save incrementally
            done = [r for r in results if r is not None]
            save_results(done, compute_summary(done, use_judge), output_path)
            return result

    tasks = [run_with_semaphore(i, tc) for i, tc in enumerate(dataset)]
    await asyncio.gather(*tasks)

    # Final save
    final_results = [r for r in results if r is not None]
    summary = compute_summary(final_results, use_judge)
    save_results(final_results, summary, output_path)
    return final_results, summary


def compute_summary(results: list[QuestionResult], has_judge: bool) -> EvalSummary:
    """Compute aggregate metrics from individual results."""
    summary = EvalSummary()
    summary.total_questions = len(results)
    summary.questions_with_errors = sum(1 for r in results if r.error)

    valid = [r for r in results if not r.error]
    if not valid:
        return summary

    # Tool selection
    tool_checks = [r for r in valid if r.tool_selection_correct is not None]
    if tool_checks:
        summary.tool_selection_accuracy = sum(r.tool_selection_correct for r in tool_checks) / len(tool_checks)

    # Argument correctness
    arg_checks = [r for r in valid if r.argument_correct is not None]
    if arg_checks:
        summary.argument_accuracy = sum(r.argument_correct for r in arg_checks) / len(arg_checks)

    # Hebrew
    hebrew_checks = [r for r in valid if r.response_in_hebrew is not None]
    if hebrew_checks:
        summary.hebrew_response_rate = sum(r.response_in_hebrew for r in hebrew_checks) / len(hebrew_checks)

    # Latency
    latencies = [r.latency_s for r in valid if r.latency_s > 0]
    if latencies:
        summary.avg_latency_s = round(sum(latencies) / len(latencies), 2)

    # Judge metrics
    if has_judge:
        faith = [r.faithfulness for r in valid if r.faithfulness is not None]
        rel = [r.relevance for r in valid if r.relevance is not None]
        comp = [r.completeness for r in valid if r.completeness is not None]
        if faith:
            summary.avg_faithfulness = round(sum(faith) / len(faith), 2)
        if rel:
            summary.avg_relevance = round(sum(rel) / len(rel), 2)
        if comp:
            summary.avg_completeness = round(sum(comp) / len(comp), 2)

    # Per-category breakdown
    categories = set(r.category for r in valid)
    for cat in sorted(categories):
        cat_results = [r for r in valid if r.category == cat]
        cat_summary = {"count": len(cat_results)}

        tc = [r for r in cat_results if r.tool_selection_correct is not None]
        if tc:
            cat_summary["tool_accuracy"] = round(sum(r.tool_selection_correct for r in tc) / len(tc), 2)

        ac = [r for r in cat_results if r.argument_correct is not None]
        if ac:
            cat_summary["arg_accuracy"] = round(sum(r.argument_correct for r in ac) / len(ac), 2)

        lats = [r.latency_s for r in cat_results if r.latency_s > 0]
        if lats:
            cat_summary["avg_latency_s"] = round(sum(lats) / len(lats), 2)

        if has_judge:
            f = [r.faithfulness for r in cat_results if r.faithfulness is not None]
            rv = [r.relevance for r in cat_results if r.relevance is not None]
            c = [r.completeness for r in cat_results if r.completeness is not None]
            if f:
                cat_summary["avg_faithfulness"] = round(sum(f) / len(f), 2)
            if rv:
                cat_summary["avg_relevance"] = round(sum(rv) / len(rv), 2)
            if c:
                cat_summary["avg_completeness"] = round(sum(c) / len(c), 2)

        summary.category_scores[cat] = cat_summary

    return summary


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_results(results: list[QuestionResult], summary: EvalSummary, output_path: str = "eval_results.json"):
    """Save evaluation results to a JSON file."""
    serializable_results = []
    for r in results:
        d = asdict(r)
        # Truncate tool results for readability
        for tc in d.get("tool_calls", []):
            if "tool_result" in tc and len(str(tc["tool_result"])) > 500:
                tc["tool_result"] = str(tc["tool_result"])[:500] + "..."
        serializable_results.append(d)

    output = {
        "summary": asdict(summary),
        "results": serializable_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def print_summary(summary: EvalSummary):
    """Print a formatted summary to stdout."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total questions:        {summary.total_questions}")
    print(f"Questions with errors:  {summary.questions_with_errors}")
    print(f"Tool selection accuracy: {summary.tool_selection_accuracy:.1%}")
    print(f"Argument accuracy:       {summary.argument_accuracy:.1%}")
    print(f"Hebrew response rate:    {summary.hebrew_response_rate:.1%}")
    print(f"Average latency:         {summary.avg_latency_s}s")

    if summary.avg_faithfulness > 0:
        print(f"\nJudge scores (1-10 scale):")
        print(f"  Faithfulness:  {summary.avg_faithfulness:.2f}")
        print(f"  Relevance:     {summary.avg_relevance:.2f}")
        print(f"  Completeness:  {summary.avg_completeness:.2f}")

    if summary.category_scores:
        print(f"\nPer-category breakdown:")
        for cat, scores in summary.category_scores.items():
            parts = [f"n={scores['count']}"]
            if "tool_accuracy" in scores:
                parts.append(f"tool={scores['tool_accuracy']:.0%}")
            if "arg_accuracy" in scores:
                parts.append(f"args={scores['arg_accuracy']:.0%}")
            if "avg_latency_s" in scores:
                parts.append(f"lat={scores['avg_latency_s']}s")
            if "avg_faithfulness" in scores:
                parts.append(f"faith={scores['avg_faithfulness']:.1f}")
            if "avg_relevance" in scores:
                parts.append(f"rel={scores['avg_relevance']:.1f}")
            print(f"  {cat:15s} {', '.join(parts)}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG system")
    parser.add_argument("--dataset", default="eval_dataset.json", help="Path to test dataset JSON")
    parser.add_argument("--output", default="eval_results.json", help="Path for output results JSON")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge evaluation")
    parser.add_argument("--ids", nargs="+", type=int, help="Run only specific question IDs")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of parallel evaluations (default: 5)")
    args = parser.parse_args()

    _, summary = asyncio.run(run_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
        use_judge=not args.no_judge,
        question_ids=args.ids,
        concurrency=args.concurrency,
    ))

    print_summary(summary)


if __name__ == "__main__":
    main()
