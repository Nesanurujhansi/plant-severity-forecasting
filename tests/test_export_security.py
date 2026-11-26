"""
Phase 09: Security Audit Unit Test
Ensures zero usage of un-trusted deserialization (pickle, joblib, dill) or dynamic execution (eval, exec)
across the codebase serving & export packages (ml/, app/).
"""

import ast
import os
from pathlib import Path
import pytest


FORBIDDEN_IMPORTS = {"pickle", "cPickle", "joblib", "dill"}
FORBIDDEN_FUNCTIONS = {"eval", "exec"}


def find_python_files(directories):
    py_files = []
    for d in directories:
        p = Path(d)
        if p.exists():
            for root, _, files in os.walk(p):
                for f in files:
                    if f.endswith(".py"):
                        py_files.append(Path(root) / f)
    return py_files


def test_zero_pickle_security_audit():
    """Grep Python source ASTs in ml/ and app/ for forbidden imports or calls."""
    target_dirs = ["ml", "app"]
    py_files = find_python_files(target_dirs)
    violations = []

    for file_path in py_files:
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        except Exception as e:
            violations.append(f"Failed to parse AST for '{file_path}': {e}")
            continue

        for node in ast.walk(tree):
            # Check import x
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in FORBIDDEN_IMPORTS:
                        violations.append(f"Forbidden import '{alias.name}' at {file_path}:{node.lineno}")
            # Check from x import y
            elif isinstance(node, ast.ImportFrom):
                if node.module in FORBIDDEN_IMPORTS:
                    violations.append(f"Forbidden import from '{node.module}' at {file_path}:{node.lineno}")
            # Check eval() / exec()
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_FUNCTIONS:
                    violations.append(f"Forbidden function call '{node.func.id}()' at {file_path}:{node.lineno}")

    assert len(violations) == 0, "Security Audit Violations Found:\n" + "\n".join(violations)
