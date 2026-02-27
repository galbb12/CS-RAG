import json
from typing import Optional

from langchain_core.tools import tool

from format_data import MOED_LABEL, SEMESTER_LABEL, format_hebrew_year


class GradesTool:
    """Tool for querying course grade distributions."""

    def __init__(self, grades_data: dict):
        self.grades = grades_data  # {course_name: [entries]}

    def to_langchain_tool(self):
        """Return a LangChain tool the LLM can call."""
        course_names = sorted(self.grades.keys())

        @tool
        def course_grades(
            course_name: str,
            lecturer: Optional[str] = None,
            year: Optional[str] = None,
            semester: Optional[str] = None,
            moed: Optional[str] = None,
            last_n: Optional[int] = None,
        ) -> str:
            """חפש התפלגות ציונים של קורס - מציג ממוצע, מספר נבחנים והיסטוגרמה לפי מועדים.
            Use this tool when the user asks about grades, averages, exam results,
            or grade distributions for a specific course.
            Args:
                course_name: שם הקורס
                lecturer: סינון לפי מרצה (אופציונלי)
                year: סינון לפי שנה עברית, למשל תשפד (אופציונלי)
                semester: סינון לפי סמסטר: A/B/C (אופציונלי)
                moed: סינון לפי מועד: a/b/c (אופציונלי)
                last_n: הצג רק N מועדים אחרונים (אופציונלי, ברירת מחדל: הכל)
            """
            # Fuzzy match
            if course_name in self.grades:
                matched = course_name
            else:
                matches = [c for c in course_names if course_name in c or c in course_name]
                if len(matches) == 1:
                    matched = matches[0]
                elif matches:
                    return f"נמצאו כמה קורסים תואמים: {', '.join(matches)}\nנא לציין את שם הקורס המדויק."
                else:
                    return f"לא נמצאו ציונים עבור: {course_name}\nקורסים זמינים: {', '.join(course_names)}"

            entries = self.grades[matched]

            # Apply filters
            if lecturer:
                filtered = [e for e in entries if lecturer in e["lecturer"] or e["lecturer"] in lecturer]
                if filtered:
                    entries = filtered
            if year:
                filtered = [e for e in entries if e["year"] == year]
                if filtered:
                    entries = filtered
            if semester:
                filtered = [e for e in entries if e["semester"] == semester.upper()]
                if filtered:
                    entries = filtered
            if moed:
                filtered = [e for e in entries if e["moed"] == moed.lower()]
                if filtered:
                    entries = filtered

            # Sort by year, semester, moed
            entries = sorted(entries, key=lambda e: (e["year"], e["semester"], e["moed"]))

            # Limit to last N if requested
            if last_n and last_n > 0 and len(entries) > last_n:
                entries = entries[-last_n:]

            # Build text summary for LLM
            lines = [f"ציונים עבור: {matched} ({len(entries)} מועדים)"]
            histogram_entries = []

            for e in entries:
                sem = SEMESTER_LABEL.get(e["semester"], e["semester"])
                moed_label = MOED_LABEL.get(e["moed"], e["moed"])
                label = f"{format_hebrew_year(e['year'])} {sem} מועד {moed_label}"
                if e["lecturer"]:
                    label += f" | {e['lecturer']}"
                if e["proj"]:
                    label += f" ({e['proj']})"

                lines.append(f"\n{label}:")
                lines.append(f"  ממוצע: {e['avg']:.1f}, נבחנים: {e['num']}")

                if e["buckets"] and len(e["buckets"]) == 10:
                    ranges = ["0-9", "10-19", "20-29", "30-39", "40-49",
                              "50-59", "60-69", "70-79", "80-89", "90-100"]
                    dist = ", ".join(f"{r}: {c}" for r, c in zip(ranges, e["buckets"]) if c > 0)
                    lines.append(f"  התפלגות: {dist}")

                    histogram_entries.append({
                        "label": label,
                        "avg": round(e["avg"], 1),
                        "num": e["num"],
                        "buckets": e["buckets"],
                    })

            text = "\n".join(lines)

            # Single histogram block for the filtered entries
            if histogram_entries:
                histogram_json = json.dumps(
                    {"course": matched, "entries": histogram_entries},
                    ensure_ascii=False,
                )
                text += f"\n\n```histogram\n{histogram_json}\n```"

            return text

        # Patch schema to add enum for course names
        props = course_grades.args_schema.model_json_schema()["properties"]
        if course_names:
            props["course_name"]["enum"] = course_names

        return course_grades
