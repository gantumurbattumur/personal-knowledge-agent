from __future__ import annotations

import subprocess
from pathlib import Path

from .schemas import WorkflowContext


def _run(command: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(command, cwd=str(cwd), check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return None


def git_branch(cwd: Path) -> str | None:
    return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)


def recent_project_files(cwd: Path, limit: int = 10) -> list[str]:
    files = [p for p in cwd.rglob("*") if p.is_file() and ".git" not in p.parts]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    relative_paths: list[str] = []
    for path in files:
        try:
            relative_paths.append(str(path.relative_to(cwd)))
        except ValueError:
            relative_paths.append(str(path))
        if len(relative_paths) >= limit:
            break
    return relative_paths


def collect_workflow_context(cwd: Path, recent_limit: int = 10) -> WorkflowContext:
    return WorkflowContext(
        cwd=str(cwd),
        git_branch=git_branch(cwd),
        recent_files=recent_project_files(cwd, limit=recent_limit),
    )
