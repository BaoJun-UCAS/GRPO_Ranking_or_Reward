"""Fast, dependency-free checks for the source checkout."""

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_parse():
    """Catch syntax errors without importing optional ML dependencies."""
    source_roots = ("src", "evaluate", "generate", "preprocess_data")
    python_files = [
        path
        for root in source_roots
        for path in (PROJECT_ROOT / root).rglob("*.py")
    ]

    assert python_files, "No Python source files were found"
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_required_project_files_exist():
    required_files = ("README.md", "LICENSE", "setup.py", "setup.cfg", "Makefile")

    for filename in required_files:
        assert (PROJECT_ROOT / filename).is_file(), f"Missing required file: {filename}"
