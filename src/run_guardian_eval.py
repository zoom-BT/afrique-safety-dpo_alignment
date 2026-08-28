"""Entry point: score a model on the UbuntuGuard guardian task and emit the report.

Run:
    python -m src.run_guardian_eval --model McGill-NLP/AfriqueQwen3.5-4B-50Langs
    python -m src.run_guardian_eval --model Qwen/Qwen3.5-4B-Base --english-control

Everything except `generate_verdicts` is pure and unit-tested, so the report assembly can
be exercised without a GPU; only generation needs one. The report deliberately carries the
instrument's own reliability alongside the score — see the vault's
`Judge_Validation_Protocol.md`, whose standing rule is that no claim may be stated more
precisely than the judge's measured reliability supports.
"""

import argparse
import json
from pathlib import Path

import yaml

from src.data import (
    build_guardian_pairs,
    filter_by_axis,
    load_ubuntuguard_rows,
    split_three_way,
)
from src.metrics import (
    UNKNOWN,
    compute_classification_metrics,
    extract_verdict,
    majority_baseline,
    per_group_metrics,
)

# Measured on Kaggle, not assumed: 64 produced a 100% unparseable rate on *both*
# backbones, and the raw completions showed why. Neither model answers directly -- both
# open with a reasoning trace ("Okay, let me try to figure out if the agent's...") and
# were cut off long before reaching the answer block. A truncated-generation artefact,
# not a failure to follow the format. 512 leaves the reasoning room to finish;
# --no-thinking is the cheaper alternative where the chat template supports it.
MAX_NEW_TOKENS = 512


def render_prompt(messages: list[dict], tokenizer, thinking: bool = True) -> str:
    """Turn chat messages into the string a model actually sees.

    Both backbones turned out to carry a ChatML template, so the fallback below is not
    the path taken in practice. It stays because a checkpoint without one would abort a
    run halfway. `thinking=False` asks the template to suppress the reasoning trace,
    which is what made a 64-token budget yield nothing parseable.

    The fallback is deliberately plain: inventing special tokens the model never saw in
    pre-training would be worse than emitting none.
    """
    if getattr(tokenizer, "chat_template", None):
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if not thinking:
            # Qwen3-family templates accept this; others reject it, so fall back rather
            # than abort a run over it.
            try:
                return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
            except TypeError:
                pass
        return tokenizer.apply_chat_template(messages, **kwargs)
    body = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
    return f"{body}\n\nassistant:"


def build_eval_records(pairs: list[dict], completions: list[str]) -> list[dict]:
    """Pair each evaluation example with the model's raw output and both parsed verdicts.

    Both extraction modes are stored per row rather than only the headline one, so the gap
    between them can be reported without re-running generation — it bounds how much of a
    score comes from the verdict parser rather than from the model.
    """
    if len(pairs) != len(completions):
        raise ValueError("expected exactly one completion per evaluation pair")
    return [
        {
            "row_id": pair["row_id"],
            "language": pair["language"],
            "theme": pair["theme"],
            "label": pair["label"],
            "completion": completion,
            "predicted": extract_verdict(completion),
            "predicted_strict": extract_verdict(completion, strict=True),
        }
        for pair, completion in zip(pairs, completions)
    ]


def build_report(records: list[dict], label: str = "african") -> dict:
    """Assemble the scored report from evaluation records. Pure, so it is unit-testable.

    Reports the majority-class floor next to the score because a macro F1 quoted without it
    is uninterpretable, and the per-language breakdown because it decides which languages
    the write-up is allowed to make claims about.
    """
    truths = [r["label"] for r in records]
    loose = compute_classification_metrics(truths, [r["predicted"] for r in records])
    strict = compute_classification_metrics(
        truths, [r["predicted_strict"] for r in records]
    )
    return {
        "slice": label,
        "n": len(records),
        "loose": loose,
        "strict": strict,
        "extractor_gap_macro_f1": loose["macro_f1"] - strict["macro_f1"],
        "majority_baseline": majority_baseline(truths),
        "per_language": per_group_metrics(records, "language"),
        "per_theme": per_group_metrics(records, "theme"),
    }


def render_report(report: dict) -> str:
    """Format a report for the terminal, flagging what must not be quoted as evidence."""
    loose, strict, floor = report["loose"], report["strict"], report["majority_baseline"]
    lines = [
        f"=== guardian task | slice: {report['slice']} | n = {report['n']} ===",
        f"  macro F1 (loose extractor) : {loose['macro_f1']:.4f}",
        f"  macro F1 (strict extractor): {strict['macro_f1']:.4f}"
        f"   [gap {report['extractor_gap_macro_f1']:+.4f} = parser artefact]",
        f"  accuracy                   : {loose['accuracy']:.4f}",
        f"  unanswered (loose/strict)  : {loose['unknown_rate']:.1%} / {strict['unknown_rate']:.1%}",
        f"  majority floor             : {floor['macro_f1']:.4f} macro F1"
        f" / {floor['accuracy']:.1%} acc (always {floor['always_predicts']})",
    ]
    verdict = "ABOVE" if loose["macro_f1"] > floor["macro_f1"] else "AT OR BELOW"
    lines.append(f"  -> {verdict} the floor")

    for title, key in (("by language", "per_language"), ("by theme", "per_theme")):
        lines.append(f"\n  {title}:")
        for group, metrics in report[key].items():
            flag = "  <- underpowered, do not quote" if metrics["underpowered"] else ""
            lines.append(
                f"    {group:<34} macro F1 {metrics['macro_f1']:.3f}"
                f"  (n={metrics['n']}){flag}"
            )
    return "\n".join(lines)


def generate_verdicts(
    pairs: list[dict],
    model,
    tokenizer,
    batch_size: int = 8,
    max_new_tokens: int = MAX_NEW_TOKENS,
    thinking: bool = True,
) -> list[str]:
    """Generate one completion per evaluation pair. The only part that needs a GPU.

    Greedy decoding (`do_sample=False`): the task has one right answer and the authors'
    own script uses `temperature=0.0`, so sampling would add variance to a measurement
    without adding anything else.
    """
    from src.evaluate import generate_batch

    prompts = [
        render_prompt(pair["prompt"], tokenizer, thinking=thinking) for pair in pairs
    ]
    completions = []
    for start in range(0, len(prompts), batch_size):
        completions.extend(
            generate_batch(
                model,
                tokenizer,
                prompts[start : start + batch_size],
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        )
    return completions


def load_eval_pairs(config: dict, english_control: bool = False) -> list[dict]:
    """Select the evaluation slice: the held-out African split, or the English controls.

    The English path is not simply "the English file". It keeps only questions whose
    `base_stem` appears in **no** African file, so nothing about them can have leaked
    through training on African data. Those 337 questions are readable by the team, which
    makes them the one slice where the judge's verdicts can be checked by hand.
    """
    dpo = config["dpo"]
    directory = Path(dpo["ubuntuguard_path"]).parent

    african_rows = load_ubuntuguard_rows(dpo["ubuntuguard_path"])
    pairs = build_guardian_pairs(african_rows)
    if dpo.get("axis"):
        pairs = filter_by_axis(pairs, dpo["axis"])
    _, agent_train, evaluation = split_three_way(
        pairs,
        eval_fraction=dpo.get("eval_fraction", 0.2),
        judge_fraction=dpo.get("judge_fraction", 0.25),
    )

    if not english_control:
        return evaluation

    english_rows = load_ubuntuguard_rows(
        directory / "Ubuntu_guard_test_all_english_only.jsonl"
    )
    english_pairs = build_guardian_pairs(english_rows)
    if dpo.get("axis"):
        # Must match the African slice's axis. The English score is only meaningful as a
        # comparison against the African one, and comparing Honest questions in one
        # language against all themes in another would put a content difference inside a
        # number that is supposed to isolate a language difference.
        english_pairs = filter_by_axis(english_pairs, dpo["axis"])

    african_stems = {p["base_stem"] for p in build_guardian_pairs(african_rows)}
    return [p for p in english_pairs if p["base_stem"] not in african_stems]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None, help="overrides config's model name")
    parser.add_argument(
        "--english-control",
        action="store_true",
        help="score the 337 contamination-free English questions instead",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
        help="generation budget; too small truncates the reasoning before the verdict",
    )
    parser.add_argument(
        "--no-thinking", action="store_true",
        help="ask the chat template to suppress the reasoning trace, where supported",
    )
    parser.add_argument("--limit", type=int, default=None, help="smoke-test on N examples")
    parser.add_argument("--output-dir", default="results/guardian")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pairs = load_eval_pairs(config, english_control=args.english_control)
    if args.limit:
        pairs = pairs[: args.limit]

    from transformers import AutoTokenizer

    from src.train import load_causal_lm
    from src.utils import set_seed

    set_seed(config["training"]["seed"])
    model_name = args.model or config["model"]["base_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_causal_lm(model_name, config["training"])
    model.eval()

    completions = generate_verdicts(
        pairs, model, tokenizer,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        thinking=not args.no_thinking,
    )
    records = build_eval_records(pairs, completions)
    report = build_report(
        records, label="english_control" if args.english_control else "african"
    )
    report["model"] = model_name

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{model_name.replace('/', '_')}_{report['slice']}"
    with open(output_dir / f"records_{tag}.jsonl", "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(output_dir / f"report_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(render_report(report))
    print(f"\nwrote {output_dir}/records_{tag}.jsonl and report_{tag}.json")
    if report["loose"]["unknown_rate"] > 0.2:
        print(
            f"\nWARNING: over 20% of outputs carried no parseable verdict at "
            f"--max-new-tokens {args.max_new_tokens}. Read the raw completions in the "
            "records file before concluding anything: if they end mid-reasoning the "
            "budget is too small rather than the model being wrong. Raise it, or pass "
            "--no-thinking to suppress the reasoning trace."
        )


if __name__ == "__main__":
    main()
