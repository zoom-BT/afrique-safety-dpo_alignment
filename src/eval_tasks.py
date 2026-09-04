"""Two more judge-free evaluations: numeric reasoning and label classification.

Both avoid asking a model to judge another model's text, which is what makes them usable by
a team that does not read the target language. `eval_mcq` covers the Honest axis; these two
cover utility and the Harmless axis.

- **AfriMGSM** (`hau`): the model works a maths problem and its answer is a number. Comparing
  numbers needs no reading. Doubles as the catastrophic-forgetting check — alignment that
  destroys arithmetic shows up here.
- **AfriHate**: classifying a tweet as Hate / Abusive / Normal is scored by log-likelihood
  over the three label words rather than by generating and parsing. Same trick as the MCQ
  evaluation, so no judge and no output-format fragility.
"""

import re

# Trailing number, tolerating thousands separators and a decimal part. Anchored at the end
# because chain-of-thought answers restate earlier figures before concluding, and the
# conclusion is what is being scored.
_NUMBER = re.compile(r"(-?\d[\d\s, ]*\.?\d*)(?!.*\d)", re.DOTALL)


def extract_final_number(text: str) -> float | None:
    """Return the last number in `text`, or None when there is none.

    The *last*, not the first: a worked solution mentions the operands before reaching the
    result, so taking the first number would score the question's own inputs.
    """
    match = _NUMBER.search(text)
    if not match:
        return None
    cleaned = re.sub(r"[\s, ]", "", match.group(1)).rstrip(".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def truncate_at(text: str, stop: list[str] | None) -> str:
    """Cut `text` at the earliest stop string, or return it unchanged.

    Needed because the models under test are *base* checkpoints: prompted few-shot, they do
    not stop after answering, they carry on inventing the next exercise. Since
    `extract_final_number` deliberately takes the last number, an untruncated completion
    would be scored on a hallucinated follow-up question rather than on the answer.
    """
    if not stop:
        return text
    cuts = [i for i in (text.find(s) for s in stop) if i >= 0]
    return text[: min(cuts)] if cuts else text


def evaluate_numeric(
    model,
    tokenizer,
    rows: list[dict],
    template: str = "{question}",
    max_new_tokens: int = 256,
    tolerance: float = 1e-6,
    stop: list[str] | None = None,
) -> dict:
    """Accuracy on rows carrying `question` and `answer_number` (AfriMGSM's schema).

    `unparsed` is reported separately from wrong answers. A model that rambles without ever
    producing a number is failing differently from one that computes badly, and collapsing
    the two would hide which.

    `stop` does double duty: it ends generation early -- a chain-of-thought answer runs to
    about fifty tokens, so stopping there rather than at `max_new_tokens` cuts the run
    several-fold -- and it truncates whatever still slipped through before scoring.
    """
    import torch

    correct = 0
    unparsed = 0
    records = []

    halting = {}
    if stop:
        from transformers.generation.stopping_criteria import (
            StoppingCriteriaList,
            StopStringCriteria,
        )

        # Construit UNE fois, hors de la boucle. StopStringCriteria indexe la totalite du
        # vocabulaire a sa construction: 5,0 s mesurees sur les 248 077 entrees de
        # Qwen3.5. Passe par `stop_strings=`, generate() le reconstruirait a chaque appel,
        # soit une fois par question -- 21 minutes de surcout pur sur 250 questions, plus
        # que la generation elle-meme.
        halting["stopping_criteria"] = StoppingCriteriaList(
            [StopStringCriteria(tokenizer=tokenizer, stop_strings=list(stop))]
        )

    for row in rows:
        prompt = template.format(question=row["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                **halting,
            )
        completion = truncate_at(
            tokenizer.decode(
                generated[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True
            ),
            stop,
        )
        predicted = extract_final_number(completion)
        expected = float(row["answer_number"])

        if predicted is None:
            unparsed += 1
        elif abs(predicted - expected) <= tolerance:
            correct += 1
        records.append(
            {"attendu": expected, "predit": predicted, "sortie": completion[:200]}
        )

    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "unparsed": unparsed,
        "unparsed_rate": unparsed / len(rows) if rows else 0.0,
        "records": records,
    }


def evaluate_classification(
    model,
    tokenizer,
    rows: list[dict],
    labels: list[str],
    text_field: str = "tweet",
    label_field: str = "label",
    template: str = "{text}\n",
) -> dict:
    """Macro F1 over a fixed label set, scored by log-likelihood rather than generation.

    Reuses the MCQ scorer: each label is treated as a candidate completion and the most
    likely one wins. That removes the whole class of failures where a model knows the answer
    but phrases it in a way the parser misses — which is what made the v1 guardian task's
    verdict extraction so treacherous.

    Macro F1 rather than accuracy: AfriHate is imbalanced, and a model answering "Normal"
    everywhere would look respectable on accuracy alone.
    """
    from src.eval_mcq import score_question

    lowered = [label.lower() for label in labels]
    confusion = {(t, p): 0 for t in range(len(labels)) for p in range(len(labels))}
    skipped = 0
    # Justesse ligne par ligne, conservee pour le test apparie. Deux bras classent les
    # memes lignes: les traiter comme des echantillons independants jetterait de la
    # puissance statistique deja payee en GPU. N'agreger que le macro F1 rendrait McNemar
    # impossible -- l'erreur commise une premiere fois sur le QCM.
    per_row = []

    for row in rows:
        gold_raw = str(row.get(label_field, "")).strip().lower()
        if gold_raw not in lowered:
            skipped += 1
            continue
        gold = lowered.index(gold_raw)
        predicted = score_question(
            model, tokenizer, row[text_field], labels, template.replace("{text}", "{question}")
        )
        confusion[(gold, predicted)] += 1
        per_row.append({"gold": gold, "predicted": predicted})

    used = sum(confusion.values())
    if used == 0:
        raise ValueError("no usable row: no label matched the expected set")

    per_label = {}
    f1_scores = []
    for i, label in enumerate(labels):
        tp = confusion[(i, i)]
        fp = sum(confusion[(t, i)] for t in range(len(labels)) if t != i)
        fn = sum(confusion[(i, p)] for p in range(len(labels)) if p != i)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "support": tp + fn}
        if tp + fn:                       # a label absent from the gold set does not count
            f1_scores.append(f1)

    return {
        "n": used,
        "skipped": skipped,
        "accuracy": sum(confusion[(i, i)] for i in range(len(labels))) / used,
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "per_label": per_label,
        "confusion": {f"{labels[t]}->{labels[p]}": c for (t, p), c in confusion.items() if c},
        "per_row": per_row,
    }


def macro_f1_from_rows(rows: list[dict], n_labels: int) -> float:
    """Macro F1 recomputed from `[{gold, predicted}]` records.

    Needed because a bootstrap resamples items, so the F1 has to be recomputed on each
    resample — the aggregate returned by `evaluate_classification` cannot be resampled.

    Labels absent from a resample's gold set are skipped, exactly as in
    `evaluate_classification`: a label nobody was supposed to predict must not drag the
    average toward zero.
    """
    f1_scores = []
    for i in range(n_labels):
        tp = sum(1 for r in rows if r["gold"] == i and r["predicted"] == i)
        fp = sum(1 for r in rows if r["gold"] != i and r["predicted"] == i)
        fn = sum(1 for r in rows if r["gold"] == i and r["predicted"] != i)
        if tp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        f1_scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def bootstrap_macro_f1(
    datasets: dict[str, list[dict]],
    statistic,
    n_labels: int,
    iterations: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict:
    """Paired bootstrap of any statistic built from several arms' macro F1 scores.

    Paired, deliberately: every arm classified the *same* items in the same order, so a
    resample must draw the same item indices for all of them. Resampling each arm
    independently would break that correspondence and inflate the interval — the same
    reasoning that makes McNemar the right test for accuracy.

    This exists because macro F1 has no closed-form paired test. The quantity the project
    actually cares about is a difference of differences — did alignment help one backbone
    more than the other — and that has no analytic variance at all.

    `statistic` receives `{name: macro_f1}` and returns one number.
    """
    import random

    noms = list(datasets)
    tailles = {len(datasets[n]) for n in noms}
    if len(tailles) != 1:
        raise ValueError(f"les bras n'ont pas le meme nombre de lignes : {tailles}")
    n = tailles.pop()
    if n == 0:
        raise ValueError("aucune ligne")

    observe = statistic({nom: macro_f1_from_rows(datasets[nom], n_labels) for nom in noms})

    alea = random.Random(seed)
    tirages = []
    for _ in range(iterations):
        indices = [alea.randrange(n) for _ in range(n)]
        tirages.append(statistic({
            nom: macro_f1_from_rows([datasets[nom][i] for i in indices], n_labels)
            for nom in noms
        }))
    tirages.sort()

    marge = (1 - confidence) / 2
    bas = tirages[int(marge * iterations)]
    haut = tirages[min(iterations - 1, int((1 - marge) * iterations))]
    # Proportion des tirages de signe oppose a l'observation, doublee: l'equivalent
    # bootstrap d'un p bilateral.
    contraires = sum(1 for t in tirages if ((t <= 0) if observe > 0 else (t >= 0)))
    return {
        "observe": observe,
        "ic_bas": bas,
        "ic_haut": haut,
        "p_bilateral": min(1.0, 2 * contraires / iterations),
        "exclut_zero": (bas > 0) or (haut < 0),
        "iterations": iterations,
    }
