"""Source provenance helpers for auditable simulator evidence."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

__all__ = ["source_provenance"]

_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def source_provenance(repo_root: str | Path) -> dict[str, str | bool]:
    """Return the exact Git commit and whether the source tree is dirty."""

    root = Path(repo_root).resolve()

    def git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"cannot determine source provenance in {root}: {exc}") from exc
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise RuntimeError(f"git returned an invalid commit identifier: {commit!r}")
    dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
    return {"git_commit": commit, "git_dirty": dirty}
