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
    """Return the dataset directory holding `marker`, e.g. `*/src/data.py`.

    Returns the first path segment under `root`, rather than counting parent levels: the
    arithmetic version was off by one and resolved to /kaggle/input itself, which broke the
    import without saying why.
    """
    if not root.exists():
        return None
    for hit in root.glob(marker):
        return root / hit.relative_to(root).parts[0]
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


def resolve_roots(cwd: Path | None = None) -> dict:
    """Return `{"code": ..., "data": ..., "output": ...}` for the current environment.

    Datasets are looked up first, the local tree second: on Kaggle both are mounted
    read-only under /kaggle/input, and anything written must go to /kaggle/working.
    """
    code = find_in_datasets("*/src/data.py") or find_locally(("src", "data.py"), cwd)
    data = find_in_datasets("*/data/Ubuntu_guard_test_crosslingual.jsonl") or find_locally(
        ("data", "Ubuntu_guard_test_crosslingual.jsonl"), cwd
    )
    if code is None:
        raise RuntimeError(
            "code not found. On Kaggle, attach the dataset holding src/ "
            "(afrique-safety-dpo-code); locally, run from inside the repository."
        )
    if data is None:
        raise RuntimeError(
            "data not found. On Kaggle, attach the dataset holding data/ "
            "(afrique-safety-dpo-data) -- it is separate from the code dataset, which is "
            "republished on every iteration and no longer carries it."
        )
    output = KAGGLE_WORKING if KAGGLE_WORKING.exists() else (code / "results")
    return {"code": code, "data": data, "output": output}
