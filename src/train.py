"""DPO training entry points built on TRL and Accelerate, plus PEFT/QLoRA helpers.

Meant to run on remote GPUs (Kaggle/Colab). Carried over from the internship's Week 1-3
training pipeline (LoRA/QLoRA/DPO groundwork already tested there) and adapted for this
topic's Native-vs-Translated DPO comparison; the dataset-loading side (formatting
UbuntuGuard's PASS/FAIL pairs into ChatML) is Week 5 work, not yet implemented here.
"""

import json
import math
import shutil
from pathlib import Path


def cleanup_checkpoint_dir(checkpoint_dir: str, keep_name: str) -> None:
    """Remove everything under `checkpoint_dir` except the entry named `keep_name`.

    Handles both files (e.g. a `README.md` `Trainer` writes alongside checkpoints)
    and directories (intermediate `checkpoint-*` snapshots) — `shutil.rmtree` alone
    fails on plain files. Safe to call more than once on the same directory.
    """
    for entry in Path(checkpoint_dir).iterdir():
        if entry.name == keep_name:
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def save_training_curves(log_history: list[dict], output_dir: str) -> dict:
    """Save training/eval loss curves (PNG) and the raw log history (JSON) to `output_dir`.

    `log_history` is `trainer.state.log_history` after `trainer.train()`: a list of dicts,
    each either a training-step entry (has `loss`) or an evaluation entry (has `eval_loss`).
    Kept separate from `run_dpo` so it can be tested without a real training run.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless backend: no display needed, safe for CI/notebooks
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_points = [
        (entry["step"], entry["loss"]) for entry in log_history if "loss" in entry
    ]
    eval_points = [
        (entry["step"], entry["eval_loss"])
        for entry in log_history
        if "eval_loss" in entry
    ]

    fig, ax = plt.subplots()
    if train_points:
        steps, losses = zip(*train_points)
        ax.plot(steps, losses, label="train_loss")
    if eval_points:
        steps, losses = zip(*eval_points)
        ax.plot(steps, losses, label="eval_loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    fig.savefig(output_path / "training_curve.png")
    plt.close(fig)

    with open(output_path / "training_log_history.json", "w") as f:
        json.dump(log_history, f, indent=2)

    return {"train_points": train_points, "eval_points": eval_points}


def build_peft_config(training_config: dict):
    """Return a `peft.LoraConfig` built from `training_config['lora']`, or `None` for full fine-tuning.

    Kept separate from the training entry points so the LoRA/full-finetune switch is
    testable without a real model.
    """
    if training_config["full_finetune"]:
        return None

    from peft import LoraConfig

    lora_config = training_config["lora"]
    return LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        task_type="CAUSAL_LM",
    )


def build_quantization_config(training_config: dict):
    """Return a `transformers.BitsAndBytesConfig` for 4-bit (QLoRA) loading, or `None`.

    Kept separate from the training entry points so the QLoRA switch is testable without a
    real model, same pattern as `build_peft_config`. NF4 (not uniform 4-bit) since network
    weights are roughly normally distributed; double quantization also compresses the
    per-block scaling constants themselves, not just the weights.
    """
    if not training_config.get("load_in_4bit", False):
        return None

    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def compute_warmup_steps(
    dataset_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    warmup_ratio: float,
    max_steps: int = -1,
) -> int:
    """Convert a warmup *ratio* into an absolute step count.

    Newer `trl` versions dropped `SFTConfig`/`DPOConfig`'s `warmup_ratio` parameter in favor
    of `warmup_steps` only, which needs the total optimizer-step count computed up front to
    stay proportional. Mirrors `Trainer`'s own rule: `max_steps` (when set) overrides
    epoch-based counting entirely rather than being combined with it.
    """
    if max_steps > 0:
        total_steps = max_steps
    else:
        effective_batch_size = batch_size * gradient_accumulation_steps
        steps_per_epoch = math.ceil(dataset_size / effective_batch_size)
        total_steps = steps_per_epoch * num_epochs
    return int(total_steps * warmup_ratio)


def build_dpo_dataset_from_pairs(pairs: list[dict]):
    """Turn a list of `{"prompt": ..., "chosen": ..., "rejected": ...}` dicts into a
    `datasets.Dataset` in the layout `trl.DPOTrainer` expects.

    Placeholder entry point for Week 5 S1: the actual UbuntuGuard PASS/FAIL -> ChatML
    formatting (and the NLLB-translated counterfactual for the Translated-DPO condition)
    is not implemented yet — this just fixes the target shape so `run_dpo` below has a
    single, stable data-loading contract regardless of where the pairs came from.
    """
    from datasets import Dataset

    return Dataset.from_list(pairs)


def run_dpo(config: dict, model_path: str | None = None, max_steps: int = -1):
    """Run DPO alignment with `trl.DPOTrainer`, using `config['dpo']` for hyperparameters.

    `model_path` (defaults to `config['model']['base_model_name']`) is the checkpoint to
    start from. Passing `ref_model=None` to `DPOTrainer` makes it create its own frozen
    copy of the starting model internally as `pi_ref`, so no separate reference-model
    checkpoint needs to be managed.
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from src.utils import get_device, set_seed

    training_config = config["training"]
    dpo_config_values = config["dpo"]
    set_seed(training_config["seed"])

    dtype_by_precision = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_by_precision[training_config["precision"]]

    model_name = model_path or config["model"]["base_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quantization_config = build_quantization_config(training_config)
    if quantization_config is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        model.to(get_device())
    else:
        # 4-bit (QLoRA) layers manage their own device placement via bitsandbytes;
        # device_map replaces the usual model.to(device) call for this path.
        from peft import prepare_model_for_kbit_training

        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, quantization_config=quantization_config, device_map={"": 0}
        )
        model = prepare_model_for_kbit_training(model)

    train_dataset = load_dataset(dpo_config_values["dataset_name"], split="train")
    train_dataset = train_dataset.select(range(dpo_config_values["train_size"]))
    eval_dataset = load_dataset(dpo_config_values["dataset_name"], split="test")
    eval_dataset = eval_dataset.select(range(dpo_config_values["eval_size"]))

    warmup_steps = compute_warmup_steps(
        dataset_size=len(train_dataset),
        batch_size=dpo_config_values["batch_size"],
        gradient_accumulation_steps=dpo_config_values["gradient_accumulation_steps"],
        num_epochs=dpo_config_values["num_epochs"],
        warmup_ratio=dpo_config_values.get("warmup_ratio", 0.0),
        max_steps=max_steps,
    )

    dpo_args = DPOConfig(
        output_dir=config["paths"]["output_dir"] + "dpo_checkpoints",
        per_device_train_batch_size=dpo_config_values["batch_size"],
        per_device_eval_batch_size=dpo_config_values["batch_size"],
        gradient_accumulation_steps=dpo_config_values["gradient_accumulation_steps"],
        gradient_checkpointing=dpo_config_values["gradient_checkpointing"],
        beta=dpo_config_values["beta"],
        num_train_epochs=dpo_config_values["num_epochs"],
        max_steps=max_steps,
        learning_rate=dpo_config_values["learning_rate"],
        warmup_steps=warmup_steps,
        max_length=training_config["max_seq_length"],
        seed=training_config["seed"],
        report_to=training_config["logging_backend"],
        bf16=(training_config["precision"] == "bf16"),
        fp16=(training_config["precision"] == "fp16"),
        eval_strategy="steps",
        eval_steps=dpo_config_values["eval_steps"],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=build_peft_config(training_config),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(config["paths"]["output_dir"] + "dpo_checkpoints/final")
    save_training_curves(trainer.state.log_history, config["paths"]["output_dir"] + "dpo")
    return trainer
