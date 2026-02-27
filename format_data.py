"""
Parse the SQL dump into LangChain Documents and kdams data.

Can be imported (parse_documents, parse_kdams) or run as a standalone script.
"""

import os
import re

from langchain_core.documents import Document


_LOCAL_SQL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "u797529344_db_rag_proj(1).sql")
SQL_PATH = os.environ.get("DB_SQL_PATH", "/etc/secrets/db.sql" if os.path.exists("/etc/secrets/db.sql") else _LOCAL_SQL)


# ---------------------------------------------------------------------------
# Generic SQL INSERT parser
# ---------------------------------------------------------------------------

def parse_inserts(sql: str, table: str) -> list[tuple]:
    """Return a list of row-tuples for every INSERT into *table*."""
    rows: list[tuple] = []
    # Match INSERT blocks for the target table (may span many lines)
    pattern = re.compile(
        rf"INSERT\s+INTO\s+`{table}`\s*\([^)]+\)\s*VALUES\s*",
        re.IGNORECASE,
    )
    for m in pattern.finditer(sql):
        pos = m.end()
        # Walk through the values section, collecting each (...) tuple
        while pos < len(sql):
            # skip whitespace / newlines
            while pos < len(sql) and sql[pos] in (" ", "\n", "\r", "\t"):
                pos += 1
            if pos >= len(sql) or sql[pos] != "(":
                break
            row, pos = _parse_row(sql, pos)
            rows.append(row)
            # skip comma or semicolon after the row
            while pos < len(sql) and sql[pos] in (" ", "\n", "\r", "\t"):
                pos += 1
            if pos < len(sql) and sql[pos] == ",":
                pos += 1
            elif pos < len(sql) and sql[pos] == ";":
                pos += 1
                break
    return rows


def _parse_row(sql: str, pos: int) -> tuple:
    """Parse a single (...) value tuple starting at *pos*. Return (tuple, new_pos)."""
    assert sql[pos] == "("
    pos += 1
    values = []
    while True:
        # skip whitespace
        while pos < len(sql) and sql[pos] in (" ", "\n", "\r", "\t"):
            pos += 1
        if sql[pos] == ")":
            pos += 1
            return tuple(values), pos
        if sql[pos] == ",":
            pos += 1
            continue
        if sql[pos] == "'":
            val, pos = _parse_string(sql, pos)
            values.append(val)
        elif sql[pos:pos+4] == "NULL":
            values.append(None)
            pos += 4
        else:
            # numeric value
            end = pos
            while end < len(sql) and sql[end] not in (",", ")", " ", "\n"):
                end += 1
            token = sql[pos:end]
            try:
                values.append(int(token))
            except ValueError:
                try:
                    values.append(float(token))
                except ValueError:
                    values.append(token)
            pos = end


def _parse_string(sql: str, pos: int) -> tuple:
    """Parse a single-quoted SQL string starting at pos. Returns (str, new_pos)."""
    assert sql[pos] == "'"
    pos += 1
    chars = []
    while pos < len(sql):
        ch = sql[pos]
        if ch == "\\" and pos + 1 < len(sql):
            nxt = sql[pos + 1]
            if nxt == "'":
                chars.append("'")
                pos += 2
                continue
            elif nxt == "n":
                chars.append("\n")
                pos += 2
                continue
            elif nxt == "\\":
                chars.append("\\")
                pos += 2
                continue
            elif nxt == '"':
                chars.append('"')
                pos += 2
                continue
            else:
                chars.append(nxt)
                pos += 2
                continue
        if ch == "'" and pos + 1 < len(sql) and sql[pos + 1] == "'":
            chars.append("'")
            pos += 2
            continue
        if ch == "'":
            pos += 1
            return "".join(chars), pos
        chars.append(ch)
        pos += 1
    return "".join(chars), pos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TAG_MAP = {
    "duty": "חובה",
    "choose": "בחירה",
    "seminar": "סמינר",
    "choose,seminar": "בחירה/סמינר",
}


def _read_sql(sql_path: str) -> str:
    with open(sql_path, encoding="utf-8") as f:
        return f.read()


def parse_kdams(sql_path: str = SQL_PATH) -> dict:
    """Parse the Tkdams table from the SQL dump. Returns {course_name: {info...}}."""
    sql = _read_sql(sql_path)
    kdams_raw = parse_inserts(sql, "Tkdams")

    kdams = {}
    for row in kdams_raw:
        code, name, ids, pts, note, prereqs, lecturer, link, grades = row
        kdams[name] = {
            "course_code": code,
            "credit_points": pts,
            "semesters_offered": ids,  # a/b/c
            "prerequisites": prereqs,
            "current_lecturers": lecturer,
            "note": note if note else None,
        }
    return kdams


def parse_documents(sql_path: str = SQL_PATH) -> list[Document]:
    """Parse the SQL dump into LangChain Documents (one per student comment)."""
    sql = _read_sql(sql_path)

    comments_raw = parse_inserts(sql, "Tcomments")
    questions_raw = parse_inserts(sql, "Tquestions")
    kdams_raw = parse_inserts(sql, "Tkdams")

    # Tquestions: id -> {course, lecturer, tag}
    questions = {}
    for row in questions_raw:
        qid, course, lecture, tag, time, rank = row
        questions[qid] = {
            "course": course,
            "lecturer": lecture,
            "tag": tag,
        }

    # Tkdams: name -> info (for enriching comment metadata)
    kdams = {}
    for row in kdams_raw:
        code, name, ids, pts, note, prereqs, lecturer, link, grades = row
        kdams[name] = {
            "course_code": code,
            "credit_points": pts,
            "semesters_offered": ids,
            "prerequisites": prereqs,
            "current_lecturers": lecturer,
        }

    documents = []
    for row in comments_raw:
        idquestion, ref, name, content, time, rank, seen = row

        if not content or not content.strip():
            continue

        q = questions.get(idquestion, {})
        course_name = q.get("course", "")
        lecturer = q.get("lecturer", "")
        tag = q.get("tag", "")

        kdam = kdams.get(course_name, {})
        course_type = TAG_MAP.get(tag, tag)

        metadata = {
            "course_name": course_name,
            "lecturer": lecturer,
            "course_type": course_type,
            "author": name,
            "date": time,
            "rank": rank,
        }

        if kdam:
            metadata["course_code"] = kdam["course_code"]
            metadata["credit_points"] = kdam["credit_points"]
            metadata["semesters_offered"] = kdam["semesters_offered"]
            if kdam["prerequisites"]:
                metadata["prerequisites"] = kdam["prerequisites"]
            if kdam["current_lecturers"]:
                metadata["current_lecturers"] = kdam["current_lecturers"]

        documents.append(Document(page_content=content, metadata=metadata))

    return documents


SEMESTER_LABEL = {"A": "א׳", "B": "ב׳", "C": "קיץ"}
MOED_LABEL = {"a": "א", "b": "ב", "c": "קיץ"}


def format_hebrew_year(year: str) -> str:
    """Add gershayim to Hebrew year: תשפד -> תשפ״ד, תשפ -> תש״פ."""
    if len(year) <= 1:
        return year
    return year[:-1] + "״" + year[-1]


def parse_grades(sql_path: str = SQL_PATH) -> dict:
    """Parse Tgrades + TgradesSemesters into a dict keyed by course name.

    Returns: {course_name: [{"lecturer", "year", "semester", "moed", "avg", "num", "buckets": [10 ints], "proj"}]}
    """
    sql = _read_sql(sql_path)

    semesters_raw = parse_inserts(sql, "TgradesSemesters")
    grades_raw = parse_inserts(sql, "Tgrades")

    # Build semester lookup: id -> info
    semesters = {}
    for row in semesters_raw:
        sid, name, lecture, year, semester, proj = row
        semesters[sid] = {
            "course": name,
            "lecturer": lecture,
            "year": year,
            "semester": semester,
            "proj": proj,
        }

    # Join grades with semesters and group by course
    result: dict[str, list] = {}
    for row in grades_raw:
        idsemester, moed, avg, num, grades_str, proj2 = row
        sem = semesters.get(idsemester)
        if not sem:
            continue
        buckets = [int(x) for x in grades_str.split(",") if x.strip()] if grades_str else []
        entry = {
            "lecturer": sem["lecturer"],
            "year": sem["year"],
            "semester": sem["semester"],
            "moed": moed,
            "avg": avg,
            "num": num,
            "buckets": buckets,
            "proj": sem["proj"],
        }
        result.setdefault(sem["course"], []).append(entry)

    return result


if __name__ == "__main__":
    docs = parse_documents()
    kdams = parse_kdams()
    grades = parse_grades()
    print(f"Parsed {len(docs)} documents, {len(kdams)} courses, {len(grades)} courses with grades")
    print(f"\n--- Sample document ---\n{docs[0]}")
