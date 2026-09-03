"""Locate the code and data roots, whether running on Kaggle or locally.

This lives in `src/` rather than in a notebook cell because the notebook version caused
three failed runs in a row -- a wrong parents index, then an eagerly-evaluated fallback
that raised before the Kaggle path was even tried. Notebook cells are the one part of this
project no test covers, so the logic that decides where everything lives belongs here.
"""

from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def find_in_datasets(marker: str, root: Path = KAGGLE_INPUT) -> Path | None:
    """Return the directory holding `marker`, e.g. `src/data.py`, searched recursively.

    Recursive, and returning the marker's own parent rather than the first segment under
    `root`: Kaggle does not mount every dataset at the same depth. One kernel showed
    `/kaggle/input/<slug>/src/...`, another `/kaggle/input/datasets/<slug>/src/...`, and a
    fixed-depth glob missed the second entirely. Three runs were lost to assumptions about
    this layout; searching costs nothing and survives whatever Kaggle does next.
    """
    if not root.exists():
        return None
    name = marker.replace("*/", "").replace("*", "")
    parts = tuple(p for p in Path(name).parts if p)
    for hit in root.rglob(parts[-1]):
        if hit.parts[-len(parts):] == parts:
            return hit.parents[len(parts) - 1]
    return None


def find_locally(marker_parts: tuple[str, ...], start: Path | None = None) -> Path | None:
    """Walk up from `start` looking for a directory containing `marker_parts`.

    Returns `None` rather than raising when nothing matches. The notebook version used a
    bare `next(...)`, which raised StopIteration on Kaggle -- where the marker legitimately
    is not on the local tree -- before the dataset lookup had a chance to run.
    """
    start = start or Path.cwd()
    for candidate in [start, *start.parents]:
        if candidate.joinpath(*marker_parts).exists():
            return candidate
    return None


UBUNTUGUARD_FILE = "Ubuntu_guard_test_crosslingual.jsonl"


def find_file(name: str, roots: list) -> Path | None:
    """Search `roots` recursively for a file called `name`, returning its full path.

    Returns the *file*, not a directory onto which a layout is then appended. Kaggle
    extracts an uploaded `data.zip` without guaranteeing the original folder level
    survives, so code assuming `<dataset>/data/<file>` breaks on the flat layout and vice
    versa. Searching for the file itself removes the assumption entirely.
    """
    for root in roots:
        if root and Path(root).exists():
            for hit in Path(root).rglob(name):
                return hit
    return None


def describe(root: Path, depth: int = 2) -> str:
    """List what is actually under `root`, so a failure reports reality instead of a guess."""
    if not root.exists():
        return f"    {root} does not exist"
    lines = [
        f"    {path.relative_to(root)}{'/' if path.is_dir() else ''}"
        for path in sorted(root.rglob("*"))[:40]
        if len(path.relative_to(root).parts) <= depth
    ]
    return "\n".join(lines) or f"    {root} is empty"


def resolve_roots(cwd: Path | None = None) -> dict:
    """Return `{"code": ..., "ubuntuguard": ..., "output": ...}` for the current environment.

    Datasets are looked up first, the local tree second: on Kaggle both are mounted
    read-only under /kaggle/input, and anything written must go to /kaggle/working.
    """
    code = find_in_datasets("*/src/data.py") or find_locally(("src", "data.py"), cwd)
    if code is None:
        raise RuntimeError(
            "code not found. On Kaggle, attach the dataset holding src/ "
            f"(afrique-safety-dpo-code). Contents of {KAGGLE_INPUT}:\n"
            + describe(KAGGLE_INPUT)
        )

    here = cwd or Path.cwd()
    ubuntuguard = find_file(UBUNTUGUARD_FILE, [KAGGLE_INPUT, here, code])
    if ubuntuguard is None:
        raise RuntimeError(
            f"{UBUNTUGUARD_FILE} not found. On Kaggle, attach afrique-safety-dpo-data -- "
            "it is separate from the code dataset, which no longer carries it since the "
            f"split. Contents of {KAGGLE_INPUT}:\n" + describe(KAGGLE_INPUT)
        )

    output = KAGGLE_WORKING if KAGGLE_WORKING.exists() else (code / "results")
    return {"code": code, "ubuntuguard": ubuntuguard, "output": output}
