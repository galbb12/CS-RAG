import json
from typing import Optional

from langchain_core.tools import tool


class KdamsTool:
    """Tool for querying the prerequisite tree (עץ קדמים)."""

    def __init__(self, kdams_path: str = "kdams.json"):
        with open(kdams_path, encoding="utf-8") as f:
            self.kdams = json.load(f)
        # Build reverse map: course -> list of courses it unlocks
        self.unlocks = {}
        for name, info in self.kdams.items():
            prereqs = info.get("prerequisites")
            if prereqs:
                for prereq in prereqs.split(","):
                    self.unlocks.setdefault(prereq, []).append(name)

    def get_prerequisites(self, course_name: str) -> Optional[dict]:
        """Get direct prerequisites for a course."""
        info = self.kdams.get(course_name)
        if not info:
            return None
        return {
            "course": course_name,
            "course_code": info.get("course_code"),
            "credit_points": info.get("credit_points"),
            "prerequisites": info.get("prerequisites", "").split(",") if info.get("prerequisites") else [],
        }

    def get_prerequisite_dag(self, course_name: str) -> Optional[dict]:
        """Get the full prerequisite DAG (recursive) for a course."""
        if course_name not in self.kdams:
            return None

        dag = {}

        def _build(name, depth=0):
            if name not in self.kdams:
                return
            # Revisit node if reached at a greater depth (longer path)
            if name in dag and dag[name]["depth"] >= depth:
                return
            info = self.kdams[name]
            prereqs = info.get("prerequisites", "").split(",") if info.get("prerequisites") else []
            dag[name] = {"depth": depth, "prerequisites": prereqs}
            for p in prereqs:
                _build(p, depth + 1)

        _build(course_name)
        # Invert depth so prerequisites (foundation) are 0 and the queried course is highest
        max_depth = max(v["depth"] for v in dag.values()) if dag else 0
        for v in dag.values():
            v["depth"] = max_depth - v["depth"]
        return dag

    def get_unlocked_courses(self, course_name: str) -> list[str]:
        """Get courses that this course is a prerequisite for."""
        return self.unlocks.get(course_name, [])

    def format_prerequisite_dag(self, course_name: str) -> str:
        """Format the full prerequisite DAG as a readable string."""
        dag = self.get_prerequisite_dag(course_name)
        if not dag:
            return f"לא נמצא קורס בשם: {course_name}"

        lines = [f"עץ קדמים עבור: {course_name}"]

        max_depth = max(info["depth"] for info in dag.values())
        # Sort by depth descending (target course first, then prerequisites)
        for name, info in sorted(dag.items(), key=lambda x: x[1]["depth"], reverse=True):
            indent = "  " * (max_depth - info["depth"])
            prereqs = info["prerequisites"]
            kdam_info = self.kdams.get(name, {})
            pts = kdam_info.get("credit_points", "?")
            if info["depth"] == max_depth:
                lines.append(f"{indent}[root] {name} ({pts} נ\"ז)")
            else:
                lines.append(f"{indent}← {name} ({pts} נ\"ז)")
            if prereqs:
                lines.append(f"{indent}   קדמים: {', '.join(prereqs)}")

        # Also show what this course unlocks
        unlocks = self.get_unlocked_courses(course_name)
        if unlocks:
            lines.append(f"\nקורסים שדורשים את {course_name} כקדם:")
            for u in unlocks:
                pts = self.kdams.get(u, {}).get("credit_points", "?")
                lines.append(f"  → {u} ({pts} נ\"ז)")

        return "\n".join(lines)

    def to_langchain_tool(self):
        """Return a LangChain tool that the LLM can call."""
        course_names = sorted(self.kdams.keys())

        @tool
        def kdams_tree(course_name: str) -> str: # The actual tool: Returns prequisites and courses unlocked for a course
            """חפש עץ קדמים של קורס - מציג את כל קורסי הקדם הנדרשים ואת הקורסים שהקורס פותח.
            Use this tool when the user asks about prerequisites, required courses,
            course order, or what courses a specific course unlocks."""
            # Fuzzy match: find the best matching course name
            if course_name in self.kdams:
                return self.format_prerequisite_dag(course_name)
            # Try partial match
            matches = [c for c in course_names if course_name in c or c in course_name]
            if len(matches) == 1:
                return self.format_prerequisite_dag(matches[0])
            if matches:
                return f"נמצאו כמה קורסים תואמים: {', '.join(matches)}\nנא לציין את שם הקורס המדויק."
            return f"לא נמצא קורס בשם: {course_name}\nקורסים זמינים: {', '.join(course_names)}"

        return kdams_tree
