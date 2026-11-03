"""
Canonical GATE 2027 Syllabi (IIT Madras) for CS, DA, EC, and EE.
Provides standardized taxonomy and dense embedding prompt formatting.
"""

from typing import Dict, List, Set

GATE_CS = "GATE CS 2027"
GATE_DA = "GATE DA 2027"
GATE_EC = "GATE EC 2027"
GATE_EE = "GATE EE 2027"

SUPPORTED_EXAMS = [GATE_CS, GATE_DA, GATE_EC, GATE_EE]

# Canonical Subjects per Exam
EXAM_SUBJECTS: Dict[str, List[str]] = {
    GATE_CS: [
        "Discrete Mathematics",
        "Linear Algebra",
        "Calculus",
        "Probability and Statistics",
        "Digital Logic",
        "Computer Organization & Architecture",
        "Programming & Data Structures",
        "Algorithms",
        "Theory of Computation",
        "Compiler Design",
        "Operating System",
        "Databases",
        "Computer Networks",
    ],
    GATE_DA: [
        "Probability and Statistics",
        "Linear Algebra",
        "Calculus and Optimization",
        "Programming, Data Structures and Algorithms",
        "Database Management and Warehousing",
        "Machine Learning",
        "Artificial Intelligence",
    ],
    GATE_EC: [
        "Engineering Mathematics",
        "Networks, Signals and Systems",
        "Electronic Devices",
        "Analog Circuits",
        "Digital Circuits",
        "Control Systems",
        "Communications",
        "Electromagnetics",
    ],
    GATE_EE: [
        "Engineering Mathematics",
        "Electric Circuits",
        "Electromagnetic Fields",
        "Signals and Systems",
        "Electrical Machines",
        "Power Systems",
        "Control Systems",
        "Electrical and Electronic Measurements",
        "Analog and Digital Electronics",
        "Power Electronics",
    ],
}

def format_slip_embedding_prompt(
    exam_tag: str,
    topic: str,
    subtopic: str,
    trap_statement: str,
    fix_statement: str = ""
) -> str:
    """
    Enriches blunder statements with canonical GATE syllabus hierarchy
    for dense vector search, semantic clustering, and blindspot indexing.
    """
    exam = exam_tag or "GATE"
    subject = topic or "General"
    sub = f" > {subtopic}" if subtopic else ""
    fix = f" Conceptual Fix: {fix_statement}" if fix_statement else ""

    return f"GATE Syllabus Context: [{exam}] {subject}{sub} | Blunder Trap: {trap_statement}.{fix}"
