"""Tests des evaluations numerique et par classification, sur modele jouet et sans GPU."""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.eval_tasks import (bootstrap_macro_f1, evaluate_classification,
                            extract_final_number, macro_f1_from_rows, truncate_at)


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


def test_troncature_protege_extract_final_number_du_few_shot():
    """Le mode d'echec que la troncature existe pour empecher.

    Amorce en few-shot, un modele base ne s'arrete pas apres avoir repondu: il enchaine sur
    un exercice invente. Comme l'extraction prend le DERNIER nombre, la reponse serait
    scoree sur la question hallucinee -- ici 99 au lieu de 18.
    """
    sortie = "Amsar shine 18.\n\nTambaya: Ali yana da kwallo 99?"
    assert extract_final_number(sortie) == 99
    assert extract_final_number(truncate_at(sortie, ["\n\n"])) == 18


def test_troncature_prend_la_premiere_occurrence():
    assert truncate_at("a FIN b Tambaya c", ["Tambaya", "FIN"]) == "a "


def test_troncature_sans_arret_ne_change_rien():
    assert truncate_at("texte intact", None) == "texte intact"
    assert truncate_at("texte intact", []) == "texte intact"


def test_troncature_laisse_passer_un_arret_absent():
    assert truncate_at("Amsar shine 18.", ["\n\n"]) == "Amsar shine 18."


def test_evaluate_numeric_construit_le_critere_une_seule_fois(tiny, monkeypatch):
    """Le critere d'arret indexe tout le vocabulaire a sa construction -- 5 s mesurees sur
    les 248 077 entrees de Qwen3.5. Le construire par appel de generate() coutait 21 min
    sur 250 questions, plus que la generation. Il doit l'etre une fois pour tout le lot."""
    import transformers.generation.stopping_criteria as sc

    from src import eval_tasks

    constructions = []
    vrai = sc.StopStringCriteria

    class Compteur(vrai):
        def __init__(self, *a, **kw):
            constructions.append(1)
            super().__init__(*a, **kw)

    monkeypatch.setattr(sc, "StopStringCriteria", Compteur)

    modele, tok = tiny
    lignes = [{"question": f"q{i}", "answer_number": i} for i in range(4)]
    eval_tasks.evaluate_numeric(
        modele, tok, lignes, max_new_tokens=4, stop=["\n\n"]
    )
    assert len(constructions) == 1, f"{len(constructions)} constructions pour 4 questions"


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


def test_classification_conserve_la_justesse_ligne_par_ligne(tiny):
    """Sans per_row, aucun test apparie n'est possible sur cet axe.

    Deux bras classent les MEMES lignes: n'agreger que le macro F1 jetterait la puissance
    statistique que l'appariement apporte. C'est l'erreur deja commise sur le QCM, ou
    per_question etait supprime comme "trop volumineux".
    """
    modele, tok = tiny
    lignes = [_ligne("a", "Normal"), _ligne("b", "Hate"), _ligne("c", "???")]
    out = evaluate_classification(modele, tok, lignes, ETIQUETTES)
    assert len(out["per_row"]) == out["n"] == 2       # la ligne ecartee n'y figure pas
    assert all(set(r) == {"gold", "predicted"} for r in out["per_row"])
    justes = sum(r["gold"] == r["predicted"] for r in out["per_row"])
    assert justes == sum(
        c for cle, c in out["confusion"].items() if cle.split("->")[0] == cle.split("->")[1]
    )


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


# --- bootstrap du macro F1 ---------------------------------------------------------


def _rangees(paires):
    return [{"gold": g, "predicted": p} for g, p in paires]


def test_macro_f1_recalcule_correspond_a_un_cas_calcule_a_la_main():
    """Deux classes, une erreur dans chaque sens: F1 de 0,5 partout, donc macro 0,5."""
    rangees = _rangees([(0, 0), (0, 1), (1, 1), (1, 0)])
    assert macro_f1_from_rows(rangees, 2) == pytest.approx(0.5)


def test_macro_f1_recalcule_ignore_une_etiquette_absente():
    """Une etiquette que personne n'etait cense predire ne doit pas tirer la moyenne a zero.

    Meme regle que dans evaluate_classification, et elle compte pour le bootstrap: un
    reechantillonnage peut tres bien ne tirer aucune ligne de la classe minoritaire.
    """
    rangees = _rangees([(0, 0), (0, 0)])
    assert macro_f1_from_rows(rangees, 3) == pytest.approx(1.0)


def test_bootstrap_apparie_refuse_des_bras_de_tailles_differentes():
    """Le bootstrap doit etre APPARIE: chaque tirage prend les memes indices dans tous les
    bras. Des tailles differentes signifient que la correspondance ligne a ligne est
    rompue, et l'intervalle serait faux sans que rien ne le signale."""
    with pytest.raises(ValueError, match="meme nombre de lignes"):
        bootstrap_macro_f1(
            {"a": _rangees([(0, 0)]), "b": _rangees([(0, 0), (1, 1)])},
            lambda f: f["b"] - f["a"], 2, iterations=10,
        )


def test_bootstrap_sur_deux_bras_identiques_contient_zero():
    memes = _rangees([(0, 0), (1, 1), (0, 1), (1, 0)] * 8)
    out = bootstrap_macro_f1({"a": memes, "b": list(memes)},
                             lambda f: f["b"] - f["a"], 2, iterations=200, seed=1)
    assert out["observe"] == pytest.approx(0.0)
    assert not out["exclut_zero"]


def test_bootstrap_detecte_un_ecart_franc():
    """Un bras parfait contre un bras qui se trompe partout: l'intervalle doit exclure zero."""
    parfait = _rangees([(0, 0), (1, 1)] * 30)
    faux = _rangees([(0, 1), (1, 0)] * 30)
    out = bootstrap_macro_f1({"a": faux, "b": parfait},
                             lambda f: f["b"] - f["a"], 2, iterations=200, seed=1)
    assert out["observe"] > 0.9
    assert out["exclut_zero"]


def test_bootstrap_accepte_une_difference_de_differences():
    """La quantite qui interesse le projet: l'alignement a-t-il davantage aide un bras ?

    Aucune variance analytique n'existe pour cela -- c'est la raison d'etre du bootstrap.
    """
    a_avant = _rangees([(0, 1), (1, 0)] * 20)          # mauvais
    a_apres = _rangees([(0, 1), (1, 0)] * 20)          # inchange
    b_avant = _rangees([(0, 1), (1, 0)] * 20)          # mauvais
    b_apres = _rangees([(0, 0), (1, 1)] * 20)          # devenu parfait
    out = bootstrap_macro_f1(
        {"a0": a_avant, "a1": a_apres, "b0": b_avant, "b1": b_apres},
        lambda f: (f["b1"] - f["b0"]) - (f["a1"] - f["a0"]),
        2, iterations=200, seed=1,
    )
    assert out["observe"] > 0.9 and out["exclut_zero"]
