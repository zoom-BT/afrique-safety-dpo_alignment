"""Verifie qu'un run interrompu laisse quelque chose d'exploitable.

Tache contractuelle de la semaine 5, restee ouverte: "verify that interrupted experiments
can resume from checkpoints". Un SFT complet dure 5,55 h contre un plafond de session
Kaggle de 9 h; une coupure sans sauvegarde intermediaire perdrait tout.
"""

import json

from src.train import IncrementalMetrics, _latest_checkpoint


class FakeState:
    def __init__(self, step):
        self.global_step = step


def test_metrics_survive_a_run_that_never_finishes(tmp_path):
    """Le coeur du besoin: chaque log est durable des qu'il est produit."""
    cible = tmp_path / "metrics_live.json"
    writer = IncrementalMetrics(cible)

    writer.on_log(None, FakeState(10), None, logs={"loss": 2.4})
    writer.on_log(None, FakeState(20), None, logs={"loss": 1.9})
    # ici la session est coupee -- rien d'autre ne s'execute

    ecrit = json.loads(cible.read_text(encoding="utf-8"))
    assert [e["step"] for e in ecrit] == [10, 20]
    assert ecrit[-1]["loss"] == 1.9


def test_metrics_file_is_created_even_before_the_first_log(tmp_path):
    cible = tmp_path / "sous" / "dossier" / "metrics.json"
    IncrementalMetrics(cible)
    assert cible.parent.exists(), "le dossier doit exister avant le premier ecrit"


def test_empty_logs_are_not_recorded(tmp_path):
    cible = tmp_path / "m.json"
    writer = IncrementalMetrics(cible)
    writer.on_log(None, FakeState(1), None, logs=None)
    assert writer.history == []


def test_latest_checkpoint_picks_the_highest_step_not_the_alphabetical_last(tmp_path):
    """checkpoint-100 vient apres checkpoint-20, ce qu'un tri de chaines inverserait."""
    for n in (20, 100, 60):
        (tmp_path / f"checkpoint-{n}").mkdir()
    assert _latest_checkpoint(str(tmp_path)).endswith("checkpoint-100")


def test_latest_checkpoint_returns_none_on_a_first_run(tmp_path):
    assert _latest_checkpoint(str(tmp_path)) is None
    assert _latest_checkpoint(str(tmp_path / "inexistant")) is None


def test_latest_checkpoint_ignores_stray_files(tmp_path):
    (tmp_path / "checkpoint-5").mkdir()
    (tmp_path / "checkpoint-notes.txt").touch()
    assert _latest_checkpoint(str(tmp_path)).endswith("checkpoint-5")


def test_both_stages_save_periodically_and_keep_the_last_few():
    """Sans save_strategy, rien n'est ecrit avant la toute fin de l'entrainement."""
    import inspect

    from src import train

    for nom in ("run_sft", "run_dpo"):
        source = inspect.getsource(getattr(train, nom))
        assert 'save_strategy="steps"' in source, nom
        assert "save_total_limit" in source, nom
        assert "resume_from_checkpoint" in source, nom
