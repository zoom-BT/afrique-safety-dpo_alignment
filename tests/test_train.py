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


def test_the_effective_sft_batch_stays_at_sixteen():
    """Le lot effectif fixe la dynamique d'entrainement; batch et accumulation sont un
    compromis memoire/vitesse qui peut bouger, leur produit non.

    Une version anterieure de ce test verrouillait batch_size == 1, ce qui figeait une
    conclusion que la mesure a ensuite renversee: batch 1 donne 97,8 s/pas contre 45,3 a
    batch 2, a lot effectif identique.
    """
    import yaml

    config = yaml.safe_load(open("config.yaml"))["sft"]
    effective = config["batch_size"] * config["gradient_accumulation_steps"]
    assert effective == 16, f"lot effectif {effective}, attendu 16"


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


def test_chunked_loss_and_automatic_device_map_are_never_combined():
    """Garde l'incompatibilite, pas un choix de valeur.

    chunked_nll patche model.forward en supposant une methode liee; device_map="auto"
    l'enveloppe dans un functools.partial pour les transferts entre appareils, et le patch
    meurt sur AttributeError. Les deux sont utiles separement -- le premier evite de
    materialiser 248 044 logits par position, le second repartit le modele -- mais jamais
    ensemble.

    Une version anterieure de ce test verrouillait loss_type == "nll", ce qui figeait un
    choix que trois OOM consecutifs ont ensuite renverse.
    """
    import yaml

    config = yaml.safe_load(open("config.yaml"))
    chunked = config["sft"].get("loss_type", "chunked_nll") == "chunked_nll"
    auto_map = config["training"].get("device_map") == "auto"
    assert not (chunked and auto_map), (
        "chunked_nll avec device_map='auto' echoue a la construction du trainer"
    )


def _config_from_yaml():
    import yaml

    return yaml.safe_load(open("config.yaml"))


def test_the_real_sft_config_can_actually_be_built():
    """Construit le vrai SFTConfig depuis config.yaml, sans modele ni GPU.

    Bien plus fort qu'une inspection de noms d'arguments: cela valide aussi les types, les
    valeurs admises et les incompatibilites entre champs. Trois runs Kaggle sont morts sur
    des erreurs que cette instanciation aurait levees en une seconde ici.
    """
    from trl import SFTConfig

    c = _config_from_yaml()
    sft, training = c["sft"], c["training"]
    SFTConfig(
        output_dir="/tmp/x",
        per_device_train_batch_size=sft["batch_size"],
        gradient_accumulation_steps=sft["gradient_accumulation_steps"],
        gradient_checkpointing=sft.get("gradient_checkpointing", False),
        num_train_epochs=sft["num_epochs"],
        learning_rate=sft["learning_rate"],
        warmup_steps=10,
        max_length=sft.get("max_length", training["max_seq_length"]),
        seed=training["seed"],
        report_to=training["logging_backend"],
        fp16=True,
        logging_steps=sft.get("logging_steps", 10),
        loss_type=sft.get("loss_type", "nll"),
    )


def test_the_real_dpo_config_can_actually_be_built():
    from trl import DPOConfig

    c = _config_from_yaml()
    dpo, training = c["dpo"], c["training"]
    DPOConfig(
        output_dir="/tmp/x",
        per_device_train_batch_size=dpo["batch_size"],
        per_device_eval_batch_size=dpo["batch_size"],
        gradient_accumulation_steps=dpo["gradient_accumulation_steps"],
        gradient_checkpointing=dpo["gradient_checkpointing"],
        beta=dpo["beta"],
        num_train_epochs=dpo["num_epochs"],
        learning_rate=dpo["learning_rate"],
        warmup_steps=10,
        max_length=dpo.get("max_length", training["max_seq_length"]),
        seed=training["seed"],
        report_to=training["logging_backend"],
        fp16=True,
        eval_strategy="no",
        eval_steps=dpo["eval_steps"],
    )


def _trainer_kwargs(function_name: str) -> set[str]:
    """Collect the keyword names our code passes to a TRL config, by reading the source."""
    import ast
    import inspect

    from src import train

    tree = ast.parse(inspect.getsource(getattr(train, function_name)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "").endswith("Config"):
            return {kw.arg for kw in node.keywords if kw.arg}
    raise AssertionError(f"no *Config(...) call found in {function_name}")


def test_every_dpo_argument_exists_in_the_installed_trl():
    # max_prompt_length was passed on faith and does not exist in this TRL. The run reached
    # the DPO step -- after loading Aya and 9 GB of weights -- before saying so.
    import dataclasses

    from trl import DPOConfig

    known = {f.name for f in dataclasses.fields(DPOConfig)}
    unknown = _trainer_kwargs("run_dpo") - known
    assert not unknown, f"DPOConfig n'accepte pas: {sorted(unknown)}"


def test_every_sft_argument_exists_in_the_installed_trl():
    import dataclasses

    from trl import SFTConfig

    known = {f.name for f in dataclasses.fields(SFTConfig)}
    unknown = _trainer_kwargs("run_sft") - known
    assert not unknown, f"SFTConfig n'accepte pas: {sorted(unknown)}"


def test_load_causal_lm_passes_arguments_transformers_actually_accepts(monkeypatch):
    """Verifie les kwargs envoyes a from_pretrained, sans telecharger 9 Go.

    Meme classe de bug que le `max_prompt_length` inexistant de DPOConfig: un nom
    d'argument errone ne se manifeste qu'apres le chargement du modele, donc apres dix
    minutes de run Kaggle.
    """
    import inspect

    import transformers
    import yaml

    from src import train

    captured = {}

    class FakeModel:
        def to(self, *_):
            return self

    def fake_from_pretrained(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(fake_from_pretrained)
    )
    monkeypatch.setattr(train, "prepare_model_for_kbit_training", lambda m: m, raising=False)

    config = yaml.safe_load(open("config.yaml"))["training"]
    config = {**config, "load_in_4bit": False}   # evite le chemin peft
    train.load_causal_lm("fake/model", config)

    accepted = set(
        inspect.signature(transformers.PreTrainedModel.from_pretrained).parameters
    )
    unknown = set(captured["kwargs"]) - accepted
    # `dtype`, `quantization_config` et `device_map` passent par **kwargs cote transformers;
    # on verifie surtout qu'on n'invente pas de nom fantaisiste.
    assert unknown <= {"dtype", "quantization_config", "device_map"}, sorted(unknown)


def test_list_lora_candidates_counts_linear_leaf_names():
    """Un nom de module qui ne correspond a rien attacherait les adaptateurs a rien."""
    import torch

    from src.train import list_lora_candidates

    model = torch.nn.Module()
    model.q_proj = torch.nn.Linear(4, 4)
    model.k_proj = torch.nn.Linear(4, 4)
    model.norm = torch.nn.LayerNorm(4)
    counts = list_lora_candidates(model)
    assert counts == {"q_proj": 1, "k_proj": 1}
