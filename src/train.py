"""DPO training entry points built on TRL and Accelerate, plus PEFT/QLoRA helpers.

Meant to run on remote GPUs (Kaggle/Colab). Carried over from the internship's Week 1-3
training pipeline (LoRA/QLoRA/DPO groundwork already tested there). Preference pairs come
from `src.data`, which carves them out of UbuntuGuard's released test split.
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


# Attention projections plus the MLP, the usual LoRA surface for a Qwen-family decoder.
# PEFT ships default target modules per architecture, but has no entry for `qwen3_5` — it
# raises "Please specify `target_modules`" rather than guessing, so they are named here.
# Verify against a real checkpoint with `list_lora_candidates` below before changing them.
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


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
        target_modules=lora_config.get("target_modules") or DEFAULT_LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )


def list_lora_candidates(model) -> dict:
    """Count the linear submodules a LoRA config could target, grouped by leaf name.

    A checkpoint whose module names differ from `DEFAULT_LORA_TARGET_MODULES` would attach
    adapters to nothing and train silently at zero effect, so this exists to check the
    names against a real model rather than trusting the convention. Vision-tower modules
    show up here too when a multimodal checkpoint is loaded whole — another reason to load
    the text-only variant.
    """
    import collections

    import torch

    counts = collections.Counter(
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    )
    return dict(counts.most_common())


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


def resolve_precision(requested: str) -> str:
    """Return a precision the current GPU actually supports, downgrading bf16 where it does not.

    Kaggle's T4 is Turing (SM 7.5) and has no bf16 support; bf16 needs Ampere (SM 8.0+).
    Asking for it anyway does not fail loudly — `accelerate` upcasts the tensors to fp32
    instead, which doubles activation memory and was half of the first OOM on this project
    (the traceback pointed straight at `_convert_to_fp32`).

    Downgrades to fp16 rather than raising, so the same config runs unchanged on a T4 and on
    an A100 at the precision each one prefers.
    """
    if requested != "bf16":
        return requested

    import torch

    if not torch.cuda.is_available():
        return requested
    if torch.cuda.is_bf16_supported():
        return "bf16"
    print(
        f"note: {torch.cuda.get_device_name(0)} has no bf16 support -- using fp16. "
        "Asking for bf16 here would silently upcast activations to fp32."
    )
    return "fp16"


def load_causal_lm(model_name: str, training_config: dict, dtype=None):
    """Load a backbone for text-only causal LM use, quantized per `training_config`.

    Both backbones here are `model_type: qwen3_5`, whose checkpoints declare
    `Qwen3_5ForConditionalGeneration` and carry a `vision_config` alongside their
    `text_config` — they are multimodal. `AutoModelForCausalLM` resolves that model type to
    `Qwen3_5ForCausalLM`, the text-only variant, which is what we want: the vision tower is
    never used here and leaving it out saves the memory it would occupy.

    That mapping only exists in transformers >= 5.12; older releases do not know `qwen3_5`
    at all and fail outright, hence the pin in requirements.txt. The error message below
    exists because the default failure is an opaque one about an unrecognised
    configuration, which is a bad thing to hit for the first time inside a GPU session.
    """
    import torch
    from transformers import AutoModelForCausalLM

    from src.utils import get_device

    dtype = dtype or torch.bfloat16
    quantization_config = build_quantization_config(training_config)
    kwargs = {"dtype": dtype}
    if quantization_config is not None:
        # 4-bit layers place themselves via bitsandbytes; device_map replaces model.to().
        kwargs |= {"quantization_config": quantization_config, "device_map": {"": 0}}

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except (KeyError, ValueError) as error:
        import transformers

        raise RuntimeError(
            f"could not load {model_name} as a causal LM with transformers "
            f"{transformers.__version__}. Both backbones are model_type 'qwen3_5', which "
            "needs transformers >= 5.12.1 — on Kaggle, run `pip install -U transformers` "
            "before importing anything from this package."
        ) from error

    if quantization_config is None:
        model.to(get_device())
    else:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    return model


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


DPO_COLUMNS = ("prompt", "chosen", "rejected")


def build_dpo_dataset_from_pairs(pairs: list[dict]):
    """Turn preference-pair dicts from `src.data` into a `datasets.Dataset` for `DPOTrainer`.

    Keeps only the three columns `trl` consumes, dropping the bookkeeping fields
    (`row_id`, `language`, `domain`, `theme`) that `src.data` carries for splitting and
    per-language reporting — `DPOTrainer` would otherwise try to tokenize them.
    """
    from datasets import Dataset

    return Dataset.from_list([{k: p[k] for k in DPO_COLUMNS} for p in pairs])


def run_dpo(config: dict, model_path: str | None = None, max_steps: int = -1):
    """Run DPO alignment with `trl.DPOTrainer`, using `config['dpo']` for hyperparameters.

    `model_path` (defaults to `config['model']['base_model_name']`) is the checkpoint to
    start from. Passing `ref_model=None` to `DPOTrainer` makes it create its own frozen
    copy of the starting model internally as `pi_ref`, so no separate reference-model
    checkpoint needs to be managed.
    """
    import torch
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from src.data import build_ubuntuguard_dpo_datasets
    from src.utils import set_seed

    training_config = config["training"]
    dpo_config_values = config["dpo"]
    set_seed(training_config["seed"])

    precision = resolve_precision(training_config["precision"])
    dtype_by_precision = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_by_precision[precision]

    model_name = model_path or config["model"]["base_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_causal_lm(model_name, training_config, dtype=dtype)

    train_pairs, eval_pairs = build_ubuntuguard_dpo_datasets(
        dpo_config_values["ubuntuguard_path"], seed=training_config["seed"]
    )
    # train_size/eval_size cap the split rather than defining it, so the A1 ablation
    # (100/200/868) is a config change rather than a re-split -- re-splitting would shuffle
    # which row_ids sit in eval and make the ablation points incomparable.
    train_dataset = build_dpo_dataset_from_pairs(
        train_pairs[: dpo_config_values["train_size"]]
    )
    eval_dataset = build_dpo_dataset_from_pairs(
        eval_pairs[: dpo_config_values["eval_size"]]
    )

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
        # Truncate the prompt, never the completion. The guardian task's answer is the last
        # handful of tokens in the sequence, so a plain right-truncation would remove the
        # entire training signal while leaving a sequence that still looks well formed.
        max_prompt_length=training_config["max_seq_length"] - 64,
        seed=training_config["seed"],
        report_to=training_config["logging_backend"],
        bf16=(precision == "bf16"),
        fp16=(precision == "fp16"),
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
