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

Two things the API cannot do, both discovered the expensive way:

1. **Attaching secrets.** No field in the SDK, no CLI flag. They are ticked once per
   notebook in the editor (Add-ons > Secrets); the attachment then persists across every
   version pushed by the API. A notebook reading a secret that is not attached starts,
   reserves the GPU, and only then fails.

2. **Pushing while the editor holds a draft.** Opening the notebook in the browser — which
   attaching secrets requires — creates a draft with its own sequence number. A push then
   fails with `ConcurrencyViolation: ExpectedSequence=N, ActualSequence=N-1`. Nothing runs
   and no quota is spent, but the push is refused. Resolve it with File > Quick Save in
   the editor (which saves without re-running, so no GPU), then close the tab and push
   again.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Values the API accepts for `machine_shape`, read out of the installed kagglesdk rather
# than guessed: NvidiaTeslaT4, NvidiaTeslaP100, Tpu1VmV38.
#
# T4 over P100: Turing has usable fp16 tensor cores and works properly with bitsandbytes
# 4-bit, which is what this project's QLoRA path needs. Neither supports bf16, so
# src.train.resolve_precision downgrades to fp16 on both -- but Pascal executes it far more
# slowly.
ACCELERATORS = {
    "t4": "NvidiaTeslaT4",
    "p100": "NvidiaTeslaP100",
    "tpu": "Tpu1VmV38",
}
DEFAULT_ACCELERATOR = "t4"


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
    accelerator: str | None = None,
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

    `accelerator` is written as `machine_shape`, not as `enable_gpu`. The SDK marks
    `enable_gpu` deprecated in favour of `machine_shape`, and only the latter lets us name
    T4 rather than accepting whichever GPU Kaggle assigns.
    """
    slug = slugify(notebook)
    return {
        "id": f"{username}/{slug}",
        "title": title or Path(notebook).stem.replace("_", " "),
        "code_file": Path(notebook).name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": private,
        "enable_internet": internet,
        "dataset_sources": dataset_sources or [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": model_sources or [],
        **({"machine_shape": ACCELERATORS[accelerator]} if accelerator else {}),
    }


def _run(args: list[str]) -> int:
    print("+", " ".join(args))
    # The Kaggle CLI writes fetched logs through the console encoding. On Windows that is
    # cp1252, which cannot represent Hausa text or the box-drawing characters notebooks
    # emit -- the fetch then dies on 'charmap codec can't encode' and leaves a 0-byte log,
    # which is how a failed run looks like no run at all.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        return subprocess.call(args, env=env)
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
        accelerator=args.accelerator,
        internet=not args.no_internet,
        private=not args.public,
        model_sources=args.model or [],
        dataset_sources=args.dataset or [],
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

    command = ["kaggle", "kernels", "push", "-p", str(staging)]
    if args.accelerator:
        # Passed both in the metadata file and on the command line: the flag wins
        # server-side, and the two agreeing removes any doubt about which took effect.
        command += ["--accelerator", ACCELERATORS[args.accelerator]]
    if args.timeout:
        command += ["-t", str(args.timeout)]
    if args.dry_run:
        # Exists because verifying the command used to require running it, which
        # publishes to the account and spends quota. Checking a command should never
        # cost what the command costs.
        print(f"\n[dry-run] rien n'a ete envoye. Retirer --dry-run pour publier.")
        return 0
    return _run(command)


# Files the notebooks actually import or read. Deliberately not the whole repository:
# notebooks/ holds a 3.4 MB PDF and .git is irrelevant on the Kaggle side.
DATASET_CONTENT = ["src", "data", "config.yaml", "requirements.txt"]


def dataset(args) -> int:
    """Package the code as a private Kaggle Dataset, replacing the git clone entirely.

    Why this exists rather than cloning: a Kaggle secret cannot be attached through the
    API, and -- measured on two real runs -- an API push appears to *clear* attachments
    made in the editor, since `kernel-metadata.json` carries no secrets field and replaces
    the kernel's settings wholesale. A notebook depending on a secret is therefore not
    submittable fire-and-forget at all.

    A dataset has none of that problem: `dataset_sources` is a real metadata field the API
    honours, and the content lands read-only under /kaggle/input. No token, no manual step.
    """
    staging = Path(args.staging) / "dataset"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for name in DATASET_CONTENT:
        source = Path(name)
        if not source.exists():
            print(f"skipping {name}: not present", file=sys.stderr)
            continue
        if source.is_dir():
            shutil.copytree(
                source,
                staging / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        else:
            shutil.copy2(source, staging / name)

    slug = args.slug or "afrique-safety-dpo-code"
    metadata = {
        "title": args.title or slug,
        "id": f"{args.username}/{slug}",
        "licenses": [{"name": "unknown"}],
    }
    (staging / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    total = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file())
    print(f"staged {total / 1e6:.1f} MB in {staging} as {args.username}/{slug}")

    if args.dry_run:
        print("\n[dry-run] rien n'a ete envoye.")
        return 0

    # `create` the first time, `version` afterwards -- the API has no upsert.
    if args.create:
        command = ["kaggle", "datasets", "create", "-p", str(staging), "-r", "zip"]
    else:
        command = ["kaggle", "datasets", "version", "-p", str(staging),
                   "-r", "zip", "-m", args.message]
    return _run(command)


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
        "--username",
        default="balbinotchoutzine",
        help="Kaggle username owning the kernel (not the GitHub org, which differs)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("push", help="write metadata and submit the notebook")
    p.add_argument("notebook")
    p.add_argument("--title")
    p.add_argument(
        "--accelerator",
        nargs="?",
        const=DEFAULT_ACCELERATOR,
        choices=sorted(ACCELERATORS),
        help="request a GPU; bare --accelerator means t4, the right choice here",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="build and stage everything, print the command, but do not submit",
    )
    p.add_argument(
        "--timeout", type=int,
        help="cap the run in seconds; a hung job otherwise eats the weekly quota",
    )
    p.add_argument("--no-internet", action="store_true")
    p.add_argument("--public", action="store_true", help="override the private default")
    p.add_argument("--model", action="append", help="Kaggle model source, repeatable")
    p.add_argument(
        "--dataset", action="append",
        help="Kaggle dataset source, repeatable. Use this instead of cloning: it needs no "
             "secret, which the API cannot attach anyway",
    )
    p.add_argument(
        "--staging", default=".kaggle_staging",
        help="directory to stage the upload in; only its contents are uploaded",
    )
    p.set_defaults(func=push)

    d = sub.add_parser("dataset", help="package the code as a private Kaggle Dataset")
    d.add_argument("--slug", help="dataset slug (default afrique-safety-dpo-code)")
    d.add_argument("--title")
    d.add_argument("--create", action="store_true", help="first upload; omit to add a version")
    d.add_argument("--message", default="update", help="version message")
    d.add_argument("--staging", default=".kaggle_staging")
    d.add_argument("--dry-run", action="store_true")
    d.set_defaults(func=dataset)

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
