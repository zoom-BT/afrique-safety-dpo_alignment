"""Tests du scoring QCM par log-vraisemblance, sur un modele jouet sans GPU."""

import math

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.eval_mcq import (
    binomial_two_sided_p,
    evaluate_mcq,
    mcnemar_p,
    normalised_log_likelihood,
    score_question,
)


@pytest.fixture(scope="module")
def tiny():
    config = AutoConfig.from_pretrained(
        "gpt2", n_layer=2, n_head=2, n_embd=32, n_positions=64
    )
    model = AutoModelForCausalLM.from_config(config).eval()
    return model, AutoTokenizer.from_pretrained("gpt2")


def test_log_likelihood_is_a_mean_not_a_sum(tiny):
    """Une somme favoriserait les reponses courtes, chaque token ne pouvant que retrancher.

    Sur un jeu ou la bonne reponse est souvent la plus longue et la plus nuancee, ce biais
    serait mesure comme de l'ignorance.
    """
    model, tok = tiny
    court = normalised_log_likelihood(model, tok, "Question:", " oui")
    long = normalised_log_likelihood(model, tok, "Question:", " oui bien sur absolument")
    # Une moyenne reste dans une plage comparable; une somme aurait un ordre de grandeur
    # d'ecart proportionnel au nombre de tokens.
    assert abs(court - long) < 6, (court, long)


def test_empty_completion_scores_minus_infinity(tiny):
    model, tok = tiny
    assert normalised_log_likelihood(model, tok, "Question:", "") == float("-inf")


def test_score_question_returns_an_index_in_range(tiny):
    model, tok = tiny
    i = score_question(model, tok, "Q ?", ["a", "b", "c", "d"], "{question}")
    assert i in range(4)


def test_score_question_is_deterministic(tiny):
    model, tok = tiny
    args = (model, tok, "Q ?", ["alpha", "beta", "gamma"], "{question}")
    assert score_question(*args) == score_question(*args)


def _row(question, choices, correct):
    labels = [0] * len(choices)
    labels[correct] = 1
    return {"question": question, "mc1_targets": {"choices": choices, "labels": labels}}


def test_evaluate_reports_the_random_baseline_from_the_real_choice_counts(tiny):
    """Un score ne veut rien dire sans le plancher qu'il doit depasser, et Uhura ne
    propose pas partout le meme nombre d'options."""
    model, tok = tiny
    rows = [_row("q1", ["a", "b"], 0), _row("q2", ["a", "b", "c", "d"], 1)]
    out = evaluate_mcq(model, tok, rows)
    assert out["n"] == 2
    assert out["random_baseline"] == pytest.approx((0.5 + 0.25) / 2)


def test_evaluate_skips_rows_without_a_correct_answer(tiny):
    model, tok = tiny
    rows = [
        _row("bonne", ["a", "b"], 0),
        {"question": "cassee", "mc1_targets": {"choices": ["a", "b"], "labels": [0, 0]}},
        {"question": "vide", "mc1_targets": {}},
    ]
    out = evaluate_mcq(model, tok, rows)
    assert out["n"] == 1 and out["skipped"] == 2


def test_evaluate_refuses_an_empty_set(tiny):
    model, tok = tiny
    with pytest.raises(ValueError, match="no usable question"):
        evaluate_mcq(model, tok, [{"question": "x", "mc1_targets": {}}])


def test_chance_level_accuracy_is_not_significant():
    """202 bonnes reponses sur 809 a quatre options, c'est exactement le hasard."""
    assert binomial_two_sided_p(202, 809, 0.25) > 0.9


def test_a_clear_gain_over_chance_is_significant():
    assert binomial_two_sided_p(400, 809, 0.25) < 1e-10


def test_p_value_is_one_when_there_is_nothing_to_test():
    assert binomial_two_sided_p(0, 0, 0.25) == 1.0


# --- test apparie ------------------------------------------------------------------


def test_mcnemar_ignores_the_questions_where_both_models_agree():
    """Les accords ne disent rien de qui est meilleur; seuls les desaccords tranchent.

    Ici 500 questions justes des deux cotes et 500 fausses des deux cotes s'ajoutent aux
    memes trois desaccords: le p ne doit pas bouger d'un iota.
    """
    maigre = mcnemar_p([True, True, True], [False, False, False])
    gras = mcnemar_p(
        [True] * 3 + [True] * 500 + [False] * 500,
        [False] * 3 + [True] * 500 + [False] * 500,
    )
    assert maigre["discordant"] == gras["discordant"] == 3
    assert maigre["p"] == pytest.approx(gras["p"])


def test_mcnemar_is_more_powerful_than_the_unpaired_test():
    """La raison d'etre de cette fonction, sur un cas ou les deux tests divergent.

    Vingt questions ou seul le second modele reussit, aucune dans l'autre sens: apparie,
    c'est ecrasant. Non apparie, vingt questions sur 808 se noieraient dans le bruit --
    c'est exactement ce qui est arrive a R0.
    """
    premier = [True] * 300 + [False] * 20 + [False] * 488
    second = [True] * 300 + [True] * 20 + [False] * 488
    out = mcnemar_p(premier, second)
    assert out["only_first"] == 0 and out["only_second"] == 20
    assert out["p"] < 1e-5


def test_mcnemar_is_symmetric_in_its_arguments():
    a = [True, False, True, False, True]
    b = [False, True, True, False, False]
    droite = mcnemar_p(a, b)
    gauche = mcnemar_p(b, a)
    assert droite["p"] == pytest.approx(gauche["p"])
    assert droite["only_first"] == gauche["only_second"]


def test_mcnemar_reports_no_evidence_when_the_models_never_disagree():
    """Zero desaccord n'est pas une preuve d'egalite: c'est une absence de preuve."""
    out = mcnemar_p([True, False, True], [True, False, True])
    assert out["discordant"] == 0 and out["p"] == 1.0


def test_mcnemar_balanced_disagreements_are_not_significant():
    """Dix desaccords de chaque cote, c'est le pile ou face parfait."""
    premier = [True] * 10 + [False] * 10
    second = [False] * 10 + [True] * 10
    assert mcnemar_p(premier, second)["p"] == pytest.approx(1.0)


def test_mcnemar_refuses_lists_of_different_lengths():
    """Un decalage d'indice apparierait des questions differentes et donnerait un p faux
    sans lever la moindre erreur."""
    with pytest.raises(ValueError, match="same questions"):
        mcnemar_p([True, False], [True])
