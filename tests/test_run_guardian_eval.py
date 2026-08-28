"""Tests for the guardian evaluation entry point.

Everything here runs without a GPU or a model: generation is the only part that needs one,
and it is isolated in `generate_verdicts`. The report assembly, the prompt rendering
fallback and the English-control selection are all pure.
"""

import pytest

from src.run_guardian_eval import (
    build_eval_records,
    build_report,
    render_prompt,
    render_report,
)


class FakeTokenizer:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "TEMPLATED:" + "|".join(m["content"] for m in messages)


MESSAGES = [
    {"role": "system", "content": "you are a guardian"},
    {"role": "user", "content": "<rules>r</rules>"},
]


def test_render_prompt_uses_the_chat_template_when_there_is_one():
    assert render_prompt(MESSAGES, FakeTokenizer("a template")).startswith("TEMPLATED:")


def test_render_prompt_falls_back_for_a_base_model_without_a_template():
    # Qwen3.5-4B-Base is one of the two backbones and is not instruction-tuned; raising
    # here would abort an evaluation run halfway through.
    rendered = render_prompt(MESSAGES, FakeTokenizer(None))
    assert "system: you are a guardian" in rendered
    assert rendered.rstrip().endswith("assistant:")


def make_pairs(n, language="Hausa", theme="misinformation or disinformation"):
    return [
        {
            "row_id": f"R{i}",
            "language": language,
            "theme": theme,
            "label": "PASS" if i % 2 else "FAIL",
        }
        for i in range(n)
    ]


def test_build_eval_records_stores_both_extraction_modes():
    pairs = make_pairs(1)
    records = build_eval_records(pairs, ["it does not fail"])
    # Loose guesses FAIL off the bare word; strict declines to guess.
    assert records[0]["predicted"] == "FAIL"
    assert records[0]["predicted_strict"] == "UNKNOWN"


def test_build_eval_records_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="one completion per"):
        build_eval_records(make_pairs(2), ["only one"])


def perfect_completions(pairs):
    return [f"<answer>\n{p['label']}\n</answer>" for p in pairs]


def test_build_report_scores_a_perfect_run_and_beats_the_floor():
    pairs = make_pairs(20)
    report = build_report(build_eval_records(pairs, perfect_completions(pairs)))
    assert report["loose"]["macro_f1"] == 1.0
    assert report["loose"]["macro_f1"] > report["majority_baseline"]["macro_f1"]


def test_build_report_quantifies_the_extractor_gap():
    # Answers with no <answer> block: loose guesses from the bare word, strict abstains.
    pairs = make_pairs(20)
    records = build_eval_records(pairs, [p["label"].lower() for p in pairs])
    report = build_report(records)
    assert report["loose"]["macro_f1"] == pytest.approx(1.0)
    assert report["strict"]["macro_f1"] == 0.0
    assert report["extractor_gap_macro_f1"] == pytest.approx(1.0)


def test_build_report_breaks_down_by_language_and_theme():
    pairs = make_pairs(20, language="Swahili") + make_pairs(20, language="Zulu")
    report = build_report(build_eval_records(pairs, perfect_completions(pairs)))
    assert set(report["per_language"]) == {"Swahili", "Zulu"}
    assert "misinformation or disinformation" in report["per_theme"]


def test_report_marks_the_slice_it_scored():
    pairs = make_pairs(10)
    report = build_report(build_eval_records(pairs, perfect_completions(pairs)), "english_control")
    assert report["slice"] == "english_control"


def test_rendered_report_warns_on_groups_too_small_to_quote():
    pairs = make_pairs(40, language="Swahili") + make_pairs(6, language="Nyanja")
    text = render_report(build_report(build_eval_records(pairs, perfect_completions(pairs))))
    assert "underpowered, do not quote" in text
    assert "Nyanja" in text


def test_rendered_report_states_whether_the_floor_was_cleared():
    pairs = make_pairs(20)
    beat = render_report(build_report(build_eval_records(pairs, perfect_completions(pairs))))
    assert "-> ABOVE the floor" in beat

    # A model that always answers PASS scores exactly the floor, and must be flagged as such.
    flat = build_eval_records(pairs, ["<answer>\nPASS\n</answer>"] * len(pairs))
    assert "-> AT OR BELOW the floor" in render_report(build_report(flat))


def test_render_prompt_asks_the_template_to_suppress_thinking():
    class ThinkingTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                                enable_thinking=True):
            return f"thinking={enable_thinking}"

    assert render_prompt(MESSAGES, ThinkingTokenizer("t"), thinking=False) == "thinking=False"


def test_render_prompt_falls_back_when_the_template_rejects_enable_thinking():
    # Templates outside the Qwen3 family reject the kwarg; a run must not abort over it.
    assert render_prompt(MESSAGES, FakeTokenizer("t"), thinking=False).startswith("TEMPLATED:")
