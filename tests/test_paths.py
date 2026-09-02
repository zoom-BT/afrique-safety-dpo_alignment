"""Tests for the code/data root resolution.

This logic previously lived in a notebook cell and caused three consecutive failed Kaggle
runs. Every failure mode below is one that actually happened.
"""

import pytest

from src.paths import find_in_datasets, find_locally, resolve_roots


def test_find_in_datasets_returns_the_dataset_dir_not_its_parent(tmp_path):
    # The arithmetic version counted parent levels, was off by one, and resolved to
    # /kaggle/input itself -- sys.path then pointed one level too high.
    dataset = tmp_path / "afrique-safety-dpo-code"
    (dataset / "src").mkdir(parents=True)
    (dataset / "src" / "data.py").touch()
    assert find_in_datasets("*/src/data.py", root=tmp_path) == dataset


def test_find_in_datasets_returns_none_when_the_mount_is_absent(tmp_path):
    assert find_in_datasets("*/src/data.py", root=tmp_path / "nope") is None


def test_find_locally_returns_none_instead_of_raising(tmp_path):
    # The notebook used a bare next(...), which raised StopIteration on Kaggle -- where the
    # marker legitimately is not on the local tree -- before the dataset lookup could run.
    assert find_locally(("src", "data.py"), start=tmp_path) is None


def test_find_locally_walks_up_to_the_repository_root(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "data.py").touch()
    deep = tmp_path / "notebooks" / "sub"
    deep.mkdir(parents=True)
    assert find_locally(("src", "data.py"), start=deep) == tmp_path


def test_resolve_roots_names_the_missing_dataset(tmp_path):
    with pytest.raises(RuntimeError, match="afrique-safety-dpo-code"):
        resolve_roots(cwd=tmp_path)


def test_resolve_roots_finds_both_locally(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "data.py").touch()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "Ubuntu_guard_test_crosslingual.jsonl").touch()
    roots = resolve_roots(cwd=tmp_path)
    assert roots["code"] == tmp_path and roots["data"] == tmp_path
