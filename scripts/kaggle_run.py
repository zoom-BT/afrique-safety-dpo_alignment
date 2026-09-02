"""Push a notebook to Kaggle as a batch GPU job, without opening the editor.

Why this exists: an interactive Kaggle session keeps burning quota until a 60-minute idle
timeout, and the weekly GPU budget is 30 hours. Running four training arms by hand in the
editor wastes hours on idling alone. `kaggle kernels push` submits a batch run instead —
GPU is consumed only while the notebook actually executes.

    python scripts/kaggle_run.py push notebooks/04_sft_hausa.ipynb --gpu
    python scripts/kaggle_run.py status 04-sft-hausa
    python scripts/kaggle_run.py fetch  04-sft-hausa --out results/

Prerequisites, both on the operator's machine and never committed:
  1. pip install kaggle
  2. kaggle.json from kaggle.com/settings -> "Create New API Token"
     placed at ~/.kaggle/kaggle.json (chmod 600 on POSIX)

`kaggle.json` is already in .gitignore.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Turing (T4) over Pascal (P100): T4 has usable fp16 tensor cores and works properly with
# bitsandbytes 4-bit, which is what this project's QLoRA path needs. Neither supports bf16,
# so src.train.resolve_precision downgrades to fp16 on both -- but the P100 executes it far
# more slowly. Kaggle exposes the choice in the notebook's accelerator setting, not here.
PREFERRED_ACCELERATOR = "T4 x2"


def slugify(name: str) -> str:
    """Turn a notebook filename into a Kaggle kernel slug.

    Kaggle slugs are lowercase, alphanumeric and hyphens, 5-50 characters. Underscores and
    dots are the usual offenders coming from a filename, so they become hyphens.
    """
    stem = Path(name).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    if len(slug) < 5:
        slug = f"{slug}-kernel"
    return slug[:50]


def build_metadata(
    notebook: str,
    username: str,
    *,
    title: str | None = None,
    gpu: bool = False,
    internet: bool = True,
    private: bool = True,
    dataset_sources: list[str] | None = None,
    model_sources: list[str] | None = None,
) -> dict:
    """Build the `kernel-metadata.json` Kaggle expects beside the notebook.

    `internet` defaults to True because every source in this project is pulled from the
    Hugging Face Hub at runtime; a Kaggle model entry is a pointer to the Hub, not an
    offline copy, so a run with internet disabled cannot load the backbones at all.

    `private` defaults to True: the vault and this repository are private, and a Kaggle
    kernel is public by default.
    """
    slug = slugify(notebook)
    return {
        "id": f"{username}/{slug}",
        "title": title or Path(notebook).stem.replace("_", " "),
        "code_file": Path(notebook).name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": private,
        "enable_gpu": gpu,
        "enable_internet": internet,
        "dataset_sources": dataset_sources or [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": model_sources or [],
    }


def _run(args: list[str]) -> int:
    print("+", " ".join(args))
    try:
        return subprocess.call(args)
    except FileNotFoundError:
        print(
            "kaggle CLI not found. Install it with `pip install kaggle`, then place your "
            "token at ~/.kaggle/kaggle.json (kaggle.com/settings -> Create New API Token).",
            file=sys.stderr,
        )
        return 127


def push(args) -> int:
    notebook = Path(args.notebook)
    if not notebook.exists():
        print(f"no such notebook: {notebook}", file=sys.stderr)
        return 1

    metadata = build_metadata(
        str(notebook),
        args.username,
        title=args.title,
        gpu=args.gpu,
        internet=not args.no_internet,
        private=not args.public,
        model_sources=args.model or [],
    )
    # Stage into a directory holding only this notebook. `kaggle kernels push -p DIR`
    # uploads everything in DIR, so pushing notebooks/ directly would ship every other
    # notebook and the 3.4 MB paper PDF alongside it.
    staging = Path(args.staging) / metadata["id"].split("/")[-1]
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook, staging / notebook.name)
    (staging / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"staged {notebook.name} + kernel-metadata.json in {staging}")
    print(json.dumps(metadata, indent=2))

    if args.gpu:
        print(
            f"\nGPU enabled. Set the accelerator to {PREFERRED_ACCELERATOR} in the kernel's "
            "settings — Kaggle also offers P100, which is Pascal and markedly slower for "
            "the 4-bit + fp16 path this project uses."
        )
    return _run(["kaggle", "kernels", "push", "-p", str(staging)])


def status(args) -> int:
    return _run(["kaggle", "kernels", "status", f"{args.username}/{args.slug}"])


def fetch(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    return _run(
        ["kaggle", "kernels", "output", f"{args.username}/{args.slug}", "-p", str(out)]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--username", default="zoombt", help="Kaggle username owning the kernel"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("push", help="write metadata and submit the notebook")
    p.add_argument("notebook")
    p.add_argument("--title")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--no-internet", action="store_true")
    p.add_argument("--public", action="store_true", help="override the private default")
    p.add_argument("--model", action="append", help="Kaggle model source, repeatable")
    p.add_argument(
        "--staging", default=".kaggle_staging",
        help="directory to stage the upload in; only its contents are uploaded",
    )
    p.set_defaults(func=push)

    s = sub.add_parser("status", help="check a submitted run")
    s.add_argument("slug")
    s.set_defaults(func=status)

    f = sub.add_parser("fetch", help="download a finished run's output")
    f.add_argument("slug")
    f.add_argument("--out", default="results/kaggle")
    f.set_defaults(func=fetch)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
