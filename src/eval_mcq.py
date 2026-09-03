"""Score a model on multiple-choice questions by log-likelihood, without generating.

Why this and not generation: the team does not read Hausa, so anything requiring a judgement
on generated text needs a judge whose own reliability must then be established. A
multiple-choice question sidesteps that entirely — the model assigns a probability to each
written-out answer, and the highest one wins. What is compared is numbers, not text, so the
scoring is exactly as trustworthy in Hausa as in English.

Uhura-TruthfulQA's `*_multiple_choice` configs give `mc1_targets` with `choices` and binary
`labels`, one correct per question. That is the Honest axis of HHH, measured with no
judgement anywhere in the loop.
"""

import math


def normalised_log_likelihood(model, tokenizer, prompt: str, completion: str) -> float:
    """Mean log-probability of `completion` given `prompt`, per token.

    Normalised by length, deliberately. A raw sum favours short answers, since every extra
    token can only subtract probability — on a set where the correct answer is often the
    longest and most qualified one, that bias would be measured as ignorance.
    """
    import torch

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    full_ids = tokenizer(prompt + completion, return_tensors="pt").input_ids
    n_prompt = prompt_ids.shape[-1]
    if full_ids.shape[-1] <= n_prompt:
        return float("-inf")      # the completion added no token

    full_ids = full_ids.to(model.device)
    with torch.no_grad():
        logits = model(full_ids).logits

    # Position i predicts token i+1, so the completion's tokens are scored by the logits
    # sitting one step to their left.
    log_probs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = full_ids[0, 1:]
    kept = log_probs[range(n_prompt - 1, len(targets)), targets[n_prompt - 1 :]]
    return kept.mean().item()


def score_question(model, tokenizer, question: str, choices: list[str], template: str) -> int:
    """Return the index of the choice the model finds most likely."""
    prompt = template.format(question=question)
    scores = [
        normalised_log_likelihood(model, tokenizer, prompt, " " + choice.strip())
        for choice in choices
    ]
    return max(range(len(scores)), key=lambda i: scores[i])


DEFAULT_TEMPLATE = "{question}"


def evaluate_mcq(model, tokenizer, rows: list[dict], template: str = DEFAULT_TEMPLATE) -> dict:
    """Accuracy over rows carrying `question` and `mc1_targets` (`choices` + `labels`).

    Reports the random baseline alongside, computed from the actual number of choices per
    question rather than assumed to be four: a score means nothing without the floor it has
    to clear, and Uhura's questions do not all offer the same number of options.
    """
    correct = 0
    usable = 0
    chance = 0.0
    per_question = []

    for row in rows:
        targets = row.get("mc1_targets") or {}
        choices = targets.get("choices") or []
        labels = targets.get("labels") or []
        if len(choices) < 2 or 1 not in labels:
            continue

        gold = labels.index(1)
        predicted = score_question(model, tokenizer, row["question"], choices, template)
        usable += 1
        chance += 1 / len(choices)
        correct += predicted == gold
        per_question.append(
            {"question": row["question"][:120], "gold": gold, "predicted": predicted}
        )

    if usable == 0:
        raise ValueError("no usable question: every row lacked choices or a correct label")

    return {
        "n": usable,
        "skipped": len(rows) - usable,
        "accuracy": correct / usable,
        "random_baseline": chance / usable,
        "correct": correct,
        "per_question": per_question,
    }


def binomial_two_sided_p(successes: int, trials: int, probability: float) -> float:
    """Probability of a result at least this extreme under chance alone.

    An accuracy above the random baseline is not evidence on its own: on 809 questions with
    four options, chance alone lands near 25% and wanders a few points either way. This
    turns "above the floor" into "further above the floor than luck explains".
    """
    if trials == 0:
        return 1.0

    def pmf(k: int) -> float:
        return (
            math.comb(trials, k)
            * probability**k
            * (1 - probability) ** (trials - k)
        )

    observed = pmf(successes)
    # Sum every outcome no more likely than the observed one -- the standard two-sided
    # construction for a distribution that is not symmetric.
    total = sum(p for k in range(trials + 1) if (p := pmf(k)) <= observed * (1 + 1e-9))
    return min(1.0, total)
