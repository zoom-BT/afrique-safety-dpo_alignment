import json

from src.train import (
    build_dpo_dataset_from_pairs,
    build_peft_config,
    build_quantization_config,
    cleanup_checkpoint_dir,
    compute_warmup_steps,
    save_training_curves,
)


def test_build_quantization_config_returns_none_when_load_in_4bit_false():
    training_config = {"load_in_4bit": False}
    assert build_quantization_config(training_config) is None


def test_build_quantization_config_returns_none_when_key_missing():
    training_config = {}
    assert build_quantization_config(training_config) is None


def test_build_quantization_config_returns_nf4_double_quant_config():
    training_config = {"load_in_4bit": True}
    quantization_config = build_quantization_config(training_config)
    assert quantization_config.load_in_4bit is True
    assert quantization_config.bnb_4bit_quant_type == "nf4"
    assert quantization_config.bnb_4bit_use_double_quant is True


def test_compute_warmup_steps_uses_max_steps_when_given():
    # max_steps overrides epoch-based counting entirely (Trainer does the same)
    assert (
        compute_warmup_steps(
            dataset_size=999999,  # irrelevant when max_steps is set
            batch_size=4,
            gradient_accumulation_steps=4,
            num_epochs=3,
            warmup_ratio=0.03,
            max_steps=1000,
        )
        == 30
    )


def test_compute_warmup_steps_computes_from_dataset_size_and_epochs():
    # effective_batch_size = 4*4 = 16; steps_per_epoch = 1600/16 = 100; total = 100*3 = 300
    assert (
        compute_warmup_steps(
            dataset_size=1600,
            batch_size=4,
            gradient_accumulation_steps=4,
            num_epochs=3,
            warmup_ratio=0.03,
        )
        == 9
    )


def test_compute_warmup_steps_rounds_steps_per_epoch_up():
    # effective_batch_size = 2; steps_per_epoch = ceil(17/2) = 9, not 8 (floor) or 8.5
    assert (
        compute_warmup_steps(
            dataset_size=17,
            batch_size=2,
            gradient_accumulation_steps=1,
            num_epochs=1,
            warmup_ratio=0.5,
        )
        == 4
    )


def test_build_peft_config_returns_none_for_full_finetune():
    training_config = {
        "full_finetune": True,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
    }
    assert build_peft_config(training_config) is None


def test_build_peft_config_returns_lora_config_with_configured_values():
    training_config = {
        "full_finetune": False,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
    }
    peft_config = build_peft_config(training_config)
    assert peft_config.r == 16
    assert peft_config.lora_alpha == 32
    assert peft_config.lora_dropout == 0.05
    assert peft_config.task_type == "CAUSAL_LM"


def test_save_training_curves_creates_png_and_json(tmp_path):
    log_history = [
        {"loss": 3.0, "step": 10},
        {"loss": 2.5, "step": 20},
        {"eval_loss": 2.8, "step": 20},
        {"loss": 2.0, "step": 30},
        {"eval_loss": 2.6, "step": 30},
    ]

    result = save_training_curves(log_history, str(tmp_path))

    assert (tmp_path / "training_curve.png").exists()
    assert (tmp_path / "training_curve.png").stat().st_size > 0
    assert (tmp_path / "training_log_history.json").exists()

    assert result["train_points"] == [(10, 3.0), (20, 2.5), (30, 2.0)]
    assert result["eval_points"] == [(20, 2.8), (30, 2.6)]


def test_save_training_curves_json_matches_input(tmp_path):
    log_history = [{"loss": 1.0, "step": 1}]
    save_training_curves(log_history, str(tmp_path))

    with open(tmp_path / "training_log_history.json") as f:
        saved = json.load(f)
    assert saved == log_history


def test_save_training_curves_handles_missing_eval_points(tmp_path):
    log_history = [{"loss": 1.0, "step": 1}, {"loss": 0.9, "step": 2}]
    result = save_training_curves(log_history, str(tmp_path))
    assert result["eval_points"] == []


def test_cleanup_checkpoint_dir_keeps_only_the_named_entry(tmp_path):
    (tmp_path / "keep.zip").write_text("zip content")
    (tmp_path / "README.md").write_text("a file, not a directory")
    checkpoint_dir = tmp_path / "checkpoint-500"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.bin").write_text("weights")
    (tmp_path / "final").mkdir()

    cleanup_checkpoint_dir(str(tmp_path), keep_name="keep.zip")

    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["keep.zip"]


def test_cleanup_checkpoint_dir_is_safe_to_rerun(tmp_path):
    (tmp_path / "keep.zip").write_text("zip content")

    cleanup_checkpoint_dir(str(tmp_path), keep_name="keep.zip")
    cleanup_checkpoint_dir(
        str(tmp_path), keep_name="keep.zip"
    )  # nothing left to clean, must not error

    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["keep.zip"]


def test_build_dpo_dataset_from_pairs_returns_dataset_with_expected_fields():
    pairs = [
        {"prompt": "p1", "chosen": "c1", "rejected": "r1"},
        {"prompt": "p2", "chosen": "c2", "rejected": "r2"},
    ]
    dataset = build_dpo_dataset_from_pairs(pairs)
    assert len(dataset) == 2
    assert dataset[0] == pairs[0]
    assert set(dataset.column_names) == {"prompt", "chosen", "rejected"}
