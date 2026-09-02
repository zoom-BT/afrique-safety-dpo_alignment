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


def test_peft_config_names_target_modules_since_peft_cannot_guess_for_qwen3_5():
    # PEFT raises "Please specify `target_modules`" for this architecture rather than
    # guessing -- this is what made the first smoke-test DPO run fail.
    config = build_peft_config(
        {"full_finetune": False, "lora": {"r": 8, "alpha": 16, "dropout": 0.0}}
    )
    assert "q_proj" in config.target_modules
    assert "down_proj" in config.target_modules


def test_peft_config_target_modules_can_be_overridden_from_config():
    config = build_peft_config(
        {
            "full_finetune": False,
            "lora": {"r": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["c_attn"]},
        }
    )
    assert list(config.target_modules) == ["c_attn"]


def test_resolve_precision_passes_through_what_it_cannot_improve():
    from src.train import resolve_precision

    assert resolve_precision("fp16") == "fp16"
    assert resolve_precision("fp32") == "fp32"


def test_resolve_precision_downgrades_bf16_when_the_gpu_lacks_it(monkeypatch):
    # Kaggle's T4 is Turing and has no bf16; asking anyway makes accelerate upcast to fp32,
    # which doubled activation memory and was half of the first OOM here.
    from src import train

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "Tesla T4")
    assert train.resolve_precision("bf16") == "fp16"


def test_resolve_precision_keeps_bf16_where_it_is_supported(monkeypatch):
    from src import train

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert train.resolve_precision("bf16") == "bf16"


def test_build_sft_dataset_keeps_only_the_columns_trl_consumes():
    # base_stem, language and friends exist for the splitter; left in place TRL would
    # tokenize them as if they were training text.
    from src.train import build_sft_dataset_from_examples

    examples = [
        {
            "prompt": [{"role": "user", "content": "Q"}],
            "completion": [{"role": "assistant", "content": "A"}],
            "base_stem": "Q",
            "language": "Hausa",
            "row_id": "aya::Hausa::Q",
        }
    ]
    ds = build_sft_dataset_from_examples(examples)
    assert set(ds.column_names) == {"prompt", "completion"}


def test_sft_and_dpo_use_different_learning_rates_by_design():
    # SFT teaches a response format, DPO nudges a preference: an SFT-sized step in DPO
    # would overwrite the behaviour SFT just installed.
    import yaml

    config = yaml.safe_load(open("config.yaml"))
    assert config["sft"]["learning_rate"] > config["dpo"]["learning_rate"]


def test_each_task_has_its_own_measured_sequence_length():
    # The 2560 in training.max_seq_length was measured on the v1 guardian task, which
    # embedded whole transcripts. Aya demonstrations run to a median of 46 tokens and the
    # Hausa DPO pairs peak at 974 -- carrying 2560 into both wasted 2.5x the memory and
    # OOM'd the first real T4 run.
    import yaml

    config = yaml.safe_load(open("config.yaml"))
    assert config["sft"]["max_length"] < config["training"]["max_seq_length"]
    assert config["dpo"]["max_length"] < config["training"]["max_seq_length"]


def test_sft_batch_is_one_because_the_vocabulary_dominates_memory():
    # 248,044 logits per position: at batch 2 and seq 2560 that is 2.5 GB before gradients.
    import yaml

    config = yaml.safe_load(open("config.yaml"))
    assert config["sft"]["batch_size"] == 1
    effective = config["sft"]["batch_size"] * config["sft"]["gradient_accumulation_steps"]
    assert effective == 16, "le lot effectif doit rester a 16"


def test_device_map_spreads_across_gpus_when_there_are_several(monkeypatch):
    # Kaggle's "T4 x2" really is two devices. The first real run OOM'd on GPU 0 while the
    # second sat idle, because the map was pinned to {"": 0} from a single-GPU assumption.
    import torch

    from src.train import resolve_device_map

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert resolve_device_map({}) == "auto"


def test_device_map_pins_to_one_device_when_there_is_only_one(monkeypatch):
    import torch

    from src.train import resolve_device_map

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert resolve_device_map({}) == {"": 0}


def test_device_map_can_be_overridden_to_keep_a_measurement_comparable():
    from src.train import resolve_device_map

    assert resolve_device_map({"device_map": {"": 0}}) == {"": 0}
