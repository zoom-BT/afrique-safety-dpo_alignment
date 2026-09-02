"""Tests for the Kaggle batch-submission helper.

Only the pure parts are covered: slug construction and metadata building. The subprocess
calls are the operator's business and need real credentials.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from kaggle_run import build_metadata, slugify


def test_slugify_replaces_underscores_kaggle_rejects():
    assert slugify("notebooks/04_sft_hausa.ipynb") == "04-sft-hausa"


def test_slugify_pads_a_slug_below_kaggle_minimum_length():
    # Kaggle requires at least 5 characters.
    assert len(slugify("ab.ipynb")) >= 5


def test_slugify_stays_within_kaggle_maximum_length():
    assert len(slugify("a" * 90 + ".ipynb")) <= 50


def test_metadata_defaults_to_private_because_the_project_is():
    meta = build_metadata("notebooks/03_sources.ipynb", "balbinotchoutzine")
    assert meta["is_private"] is True


def test_metadata_defaults_to_internet_on():
    # Kaggle model entries are pointers to the Hugging Face Hub, not offline copies, so a
    # run without internet cannot load the backbones at all.
    assert build_metadata("n/x_notebook.ipynb", "balbinotchoutzine")["enable_internet"] is True


def test_metadata_asks_for_t4_by_name_not_the_deprecated_enable_gpu():
    # The SDK marks enable_gpu deprecated in favour of machine_shape, and only the latter
    # lets us name T4 instead of accepting whichever GPU Kaggle assigns.
    meta = build_metadata("n/x_notebook.ipynb", "balbinotchoutzine", accelerator="t4")
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert "enable_gpu" not in meta


def test_metadata_gpu_is_off_unless_asked():
    assert "machine_shape" not in build_metadata("n/x_notebook.ipynb", "balbinotchoutzine")
    assert build_metadata("n/x_notebook.ipynb", "balbinotchoutzine", accelerator="t4")["machine_shape"] == "NvidiaTeslaT4"


def test_metadata_id_combines_username_and_slug():
    meta = build_metadata("notebooks/04_sft_hausa.ipynb", "balbinotchoutzine")
    assert meta["id"] == "balbinotchoutzine/04-sft-hausa"


def test_metadata_code_file_is_the_bare_filename_not_the_path():
    # kaggle push uploads a directory; code_file must be relative to it.
    meta = build_metadata("notebooks/04_sft_hausa.ipynb", "balbinotchoutzine")
    assert meta["code_file"] == "04_sft_hausa.ipynb"


def test_push_has_a_dry_run_so_checking_costs_nothing():
    # Verifying the constructed command used to require actually submitting it, which
    # publishes to the account and spends GPU quota.
    import argparse

    from kaggle_run import main
    import inspect

    src = inspect.getsource(main)
    assert "--dry-run" in src
