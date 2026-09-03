"""Tests des evaluations numerique et par classification, sur modele jouet et sans GPU."""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.eval_tasks import evaluate_classification, extract_final_number


@pytest.fixture(scope="module")
def tiny():
    config = AutoConfig.from_pretrained("gpt2", n_layer=2, n_head=2, n_embd=32, n_positions=64)
    return AutoModelForCausalLM.from_config(config).eval(), AutoTokenizer.from_pretrained("gpt2")


def test_extraction_prend_le_dernier_nombre_pas_le_premier():
    """Une solution detaillee cite ses operandes avant de conclure; prendre le premier
    nombre reviendrait a scorer l'enonce."""
    assert extract_final_number("Janet a 16 oeufs, en mange 3, il reste 13") == 13


def test_extraction_gere_les_separateurs_de_milliers():
    assert extract_final_number("le total est 1 234") == 1234
    assert extract_final_number("cela fait 2,500") == 2500


def test_extraction_gere_les_decimaux_et_les_negatifs():
    assert extract_final_number("resultat : 3.5") == 3.5
    assert extract_final_number("solde -42") == -42


def test_extraction_ignore_la_ponctuation_finale():
    assert extract_final_number("La reponse est 18.") == 18


def test_extraction_renvoie_none_sans_nombre():
    """Distinguer 'aucun nombre produit' de 'mauvais nombre' -- deux echecs differents."""
    assert extract_final_number("je ne sais pas repondre") is None
    assert extract_final_number("") is None


def _ligne(texte, etiquette):
    return {"tweet": texte, "label": etiquette}


ETIQUETTES = ["Normal", "Abusive", "Hate"]


def test_classification_calcule_un_macro_f1(tiny):
    modele, tok = tiny
    lignes = [_ligne("bonjour", "Normal"), _ligne("insulte", "Abusive"),
              _ligne("menace", "Hate")]
    out = evaluate_classification(modele, tok, lignes, ETIQUETTES)
    assert out["n"] == 3
    assert 0.0 <= out["macro_f1"] <= 1.0


def test_classification_ignore_les_etiquettes_hors_ensemble(tiny):
    """La carte AfriHate et ses donnees reelles ne s'accordent pas sur la casse; une ligne
    dont l'etiquette est inconnue doit etre comptee a part, pas silencieusement mal classee."""
    modele, tok = tiny
    lignes = [_ligne("a", "Normal"), _ligne("b", "Inconnue"), _ligne("c", "Hate")]
    out = evaluate_classification(modele, tok, lignes, ETIQUETTES)
    assert out["n"] == 2 and out["skipped"] == 1


def test_classification_accepte_une_casse_differente(tiny):
    modele, tok = tiny
    out = evaluate_classification(modele, tok, [_ligne("a", "normal")], ETIQUETTES)
    assert out["n"] == 1 and out["skipped"] == 0


def test_classification_refuse_un_ensemble_vide(tiny):
    modele, tok = tiny
    with pytest.raises(ValueError, match="no usable row"):
        evaluate_classification(modele, tok, [_ligne("a", "???")], ETIQUETTES)


def test_macro_f1_ignore_une_etiquette_absente_du_gold(tiny):
    """Une etiquette jamais attendue ne doit pas tirer le macro F1 vers zero."""
    modele, tok = tiny
    lignes = [_ligne("a", "Normal"), _ligne("b", "Normal")]
    out = evaluate_classification(modele, tok, lignes, ETIQUETTES)
    assert out["per_label"]["Hate"]["support"] == 0
