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


def git_root(cwd: Path) -> Path | None:
    root = _run(["git", "rev-parse", "--show-toplevel"], cwd)
    if not root:
        return None
    return Path(root).expanduser().resolve()


def discover_project_root(cwd: Path, prefer_git: bool = True) -> Path:
    cwd = cwd.expanduser().resolve()
    if prefer_git:
        root = git_root(cwd)
        if root:
            return root
    return cwd


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
    resolved_cwd = cwd.expanduser().resolve()
    resolved_git_root = git_root(resolved_cwd)
    project_root = discover_project_root(resolved_cwd, prefer_git=True)
    return WorkflowContext(
        cwd=str(resolved_cwd),
        project_root=str(project_root),
        git_root=str(resolved_git_root) if resolved_git_root else None,
        git_branch=git_branch(resolved_cwd),
        recent_files=recent_project_files(project_root, limit=recent_limit),
    )
