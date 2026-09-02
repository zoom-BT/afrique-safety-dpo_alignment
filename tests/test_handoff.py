"""Verifie la reprise SFT -> DPO sur un vrai adaptateur LoRA, sans GPU.

Un bras complet coute 9,76 h contre un plafond de session Kaggle de 9 h, donc le SFT et le
DPO sont soumis separement et le second doit repartir de l'adaptateur produit par le
premier. Si cette reprise echoue silencieusement, le DPO s'entraine depuis le modele base
et le resultat est faux sans qu'aucune erreur ne le signale.
"""

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM


def _tiny_model():
    """Un GPT2 minuscule: quelques milliers de parametres, aucun telechargement de poids."""
    config = AutoConfig.from_pretrained(
        "gpt2", n_layer=1, n_head=1, n_embd=8, vocab_size=32, n_positions=16
    )
    return AutoModelForCausalLM.from_config(config)


def _lora():
    return LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], task_type="CAUSAL_LM")


def test_an_adapter_saved_by_the_sft_stage_is_not_a_full_checkpoint(tmp_path):
    """C'est le coeur du probleme: save_model sur un PeftModel n'ecrit que l'adaptateur."""
    peft_model = get_peft_model(_tiny_model(), _lora())
    peft_model.save_pretrained(tmp_path)

    ecrits = {f.name for f in tmp_path.iterdir()}
    assert "adapter_config.json" in ecrits
    assert "config.json" not in ecrits, (
        "sans config.json, AutoModelForCausalLM ne peut pas lire ce dossier -- "
        "c'est pourquoi le chemin de l'adaptateur ne peut pas etre passe comme model_path"
    )


def test_the_adapter_is_reloaded_onto_the_base_and_stays_trainable(tmp_path):
    """`is_trainable=True` est ce qui separe un DPO qui apprend d'un DPO qui tourne a vide."""
    peft_model = get_peft_model(_tiny_model(), _lora())
    peft_model.save_pretrained(tmp_path)

    recharge = PeftModel.from_pretrained(_tiny_model(), tmp_path, is_trainable=True)
    entrainables = [n for n, p in recharge.named_parameters() if p.requires_grad]
    assert entrainables, "aucun parametre entrainable: le DPO n'apprendrait rien"
    assert all("lora" in n.lower() for n in entrainables), (
        "seuls les poids LoRA doivent etre entrainables"
    )


def test_loading_frozen_would_train_nothing_and_raise_no_error(tmp_path):
    """Documente le mode d'echec silencieux que `is_trainable=True` evite."""
    peft_model = get_peft_model(_tiny_model(), _lora())
    peft_model.save_pretrained(tmp_path)

    gele = PeftModel.from_pretrained(_tiny_model(), tmp_path)   # defaut: inference
    assert not [n for n, p in gele.named_parameters() if p.requires_grad], (
        "sans is_trainable, tout est gele et la loss bougerait a peine -- sans erreur"
    )


def test_the_reloaded_adapter_carries_the_trained_weights(tmp_path):
    """Reprendre le bon adaptateur, pas un adaptateur neuf."""
    peft_model = get_peft_model(_tiny_model(), _lora())
    with torch.no_grad():
        for nom, param in peft_model.named_parameters():
            if "lora_B" in nom:
                param.fill_(0.375)          # lora_B est initialise a zero
    peft_model.save_pretrained(tmp_path)

    recharge = PeftModel.from_pretrained(_tiny_model(), tmp_path, is_trainable=True)
    valeurs = [p for n, p in recharge.named_parameters() if "lora_B" in n]
    assert valeurs and all(torch.allclose(v, torch.full_like(v, 0.375)) for v in valeurs)


def test_run_dpo_skips_peft_config_when_an_adapter_is_already_loaded():
    """Empiler un second adaptateur laisserait celui du SFT gele en dessous."""
    import inspect

    from src.train import run_dpo

    source = inspect.getsource(run_dpo)
    assert "peft_config=None if adapter_path else" in source
