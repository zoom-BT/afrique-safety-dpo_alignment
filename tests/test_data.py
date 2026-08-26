"""Tests for the UbuntuGuard -> DPO preference-pair pipeline.

Fixtures are hand-written miniatures of the real rows (same field names, same `User:` /
` Agent:` markers, same Python-repr `metadata`) so the whole module is testable without the
10 MB source files.
"""

import pytest

from src.data import (
    build_pair,
    build_preference_pairs,
    parse_metadata,
    parse_transcript,
    split_by_base_stem,
)


def make_row(row_id, base_id, label, transcript, language="Swahili"):
    return {
        "policy": "1. Never endorse harmful claims.",
        "transcript": transcript,
        "label": label,
        "metadata": "{'num_rules': 7, 'violated_policies': []}",
        "row_id": row_id,
        "base_id": base_id,
        "country_code": "KEN",
        "country": "Kenya",
        "language": language,
        "theme": "specialized advice",
        "topic": "education",
        "domain": "education",
        "sensitive_characteristic": "gender",
    }


TWO_TURN_PASS = "User: First question?\n Agent: Careful answer.\nUser: And then?\n Agent: Safe follow-up."
TWO_TURN_FAIL = "User: First question?\n Agent: Careful answer.\nUser: And then?\n Agent: Harmful follow-up."
FIRST_TURN_FAIL = "User: First question?\n Agent: Reckless answer.\nUser: And then?\n Agent: Harmful follow-up."


def test_parse_metadata_reads_python_repr_not_json():
    # json.loads would raise on single quotes; this is the trap the helper exists for.
    assert parse_metadata("{'num_rules': 7, 'violated_policies': []}") == {
        "num_rules": 7,
        "violated_policies": [],
    }


def test_parse_transcript_normalises_agent_to_assistant():
    turns = parse_transcript(TWO_TURN_PASS)
    assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant"]
    assert turns[0]["content"] == "First question?"
    assert turns[-1]["content"] == "Safe follow-up."


def test_parse_transcript_handles_indented_agent_marker():
    # The real data indents `Agent:` by a space rather than starting it at column 0.
    turns = parse_transcript("User: Q\n    Agent: A")
    assert [t["role"] for t in turns] == ["user", "assistant"]


def test_parse_transcript_returns_empty_for_unmarked_text():
    assert parse_transcript("no markers at all") == []


def test_build_pair_truncates_at_the_first_divergent_response():
    # The real data diverges at turn 1 for 839 of 843 pairs: same opening question, then a
    # compliant vs. a violating answer. Everything after that point is dropped.
    pair = build_pair(
        make_row("R1", "R1_llama", "PASS", TWO_TURN_PASS),
        make_row("R1", "R1_qwen", "FAIL", FIRST_TURN_FAIL),
    )
    assert [t["role"] for t in pair["prompt"]] == ["system", "user"]
    assert pair["chosen"] == [{"role": "assistant", "content": "Careful answer."}]
    assert pair["rejected"] == [{"role": "assistant", "content": "Reckless answer."}]


def test_build_pair_handles_divergence_on_a_later_turn():
    # The remaining handful share their first response and split later; the prompt then
    # legitimately carries the earlier turns.
    pair = build_pair(
        make_row("R1", "R1_llama", "PASS", TWO_TURN_PASS),
        make_row("R1", "R1_qwen", "FAIL", TWO_TURN_FAIL),
    )
    assert [t["role"] for t in pair["prompt"]] == ["system", "user", "assistant", "user"]
    assert pair["rejected"] == [{"role": "assistant", "content": "Harmful follow-up."}]


def test_build_pair_can_omit_policy():
    pair = build_pair(
        make_row("R1", "R1_llama", "PASS", TWO_TURN_PASS),
        make_row("R1", "R1_qwen", "FAIL", FIRST_TURN_FAIL),
        include_policy=False,
    )
    assert [t["role"] for t in pair["prompt"]] == ["user"]


def test_build_pair_rejects_divergence_on_a_user_turn():
    # Differing *user* text means the two sides answer different questions -- 3 real pairs
    # look like this and must be dropped, not silently kept.
    diverging = "User: A DIFFERENT question?\n Agent: Careful answer.\nUser: And then?\n Agent: Safe follow-up."
    assert (
        build_pair(
            make_row("R1", "R1_llama", "PASS", TWO_TURN_PASS),
            make_row("R1", "R1_qwen", "FAIL", diverging),
        )
        is None
    )


def test_build_pair_rejects_identical_transcripts():
    # No divergence means no preference signal to learn from.
    assert (
        build_pair(
            make_row("R1", "R1_llama", "PASS", TWO_TURN_PASS),
            make_row("R1", "R1_qwen", "FAIL", TWO_TURN_PASS),
        )
        is None
    )


def test_build_pair_rejects_truncated_dialogue():
    # Real data contains transcripts ending on a user turn, with no response to contrast.
    truncated = "User: Q\n Agent: A\nUser: dangling"
    assert (
        build_pair(
            make_row("R1", "R1_llama", "PASS", truncated),
            make_row("R1", "R1_qwen", "FAIL", truncated),
        )
        is None
    )


def test_build_preference_pairs_matches_multiple_responses_without_reuse():
    # A row_id with two PASS and two FAIL yields two pairs, each response used exactly once.
    rows = [
        make_row("R1", "R1_a", "PASS", TWO_TURN_PASS),
        make_row("R1", "R1_b", "PASS", TWO_TURN_PASS),
        make_row("R1", "R1_c", "FAIL", TWO_TURN_FAIL),
        make_row("R1", "R1_d", "FAIL", TWO_TURN_FAIL),
    ]
    assert len(build_preference_pairs(rows)) == 2


def test_build_preference_pairs_is_limited_by_the_scarcer_label():
    # Two PASS but one FAIL yields one pair, not two -- no reusing the single FAIL.
    rows = [
        make_row("R1", "R1_a", "PASS", TWO_TURN_PASS),
        make_row("R1", "R1_b", "PASS", TWO_TURN_PASS),
        make_row("R1", "R1_c", "FAIL", TWO_TURN_FAIL),
    ]
    assert len(build_preference_pairs(rows)) == 1


def test_build_preference_pairs_is_order_independent():
    rows = [
        make_row("R1", "R1_a", "PASS", TWO_TURN_PASS),
        make_row("R2", "R2_a", "PASS", TWO_TURN_PASS),
        make_row("R1", "R1_c", "FAIL", TWO_TURN_FAIL),
        make_row("R2", "R2_c", "FAIL", TWO_TURN_FAIL),
    ]
    assert build_preference_pairs(rows) == build_preference_pairs(list(reversed(rows)))


def make_pairs(n_per_language, stem_prefix=None):
    """Build throwaway pairs. By default every pair gets its own base_stem, so a test only
    exercises cross-language sharing when it deliberately reuses a `stem_prefix`."""
    pairs = []
    for language, count in n_per_language.items():
        for i in range(count):
            stem = f"{stem_prefix}{i}" if stem_prefix else f"{language}{i}"
            pairs.append(
                {
                    "row_id": f"{language}{i}",
                    "base_stem": stem,
                    "language": language,
                    "x": i,
                }
            )
    return pairs


def test_split_by_base_stem_leaves_no_row_id_on_both_sides():
    train, evaluation = split_by_base_stem(make_pairs({"Swahili": 50, "Nyanja": 10}))
    assert {p["row_id"] for p in train} & {p["row_id"] for p in evaluation} == set()


def test_split_keeps_pairs_from_one_row_id_together():
    # Two pairs sharing a row_id share an identical prompt; separating them leaks it.
    pairs = [
        {"row_id": "R1", "base_stem": "S1", "language": "Swahili", "x": 0},
        {"row_id": "R1", "base_stem": "S1", "language": "Swahili", "x": 1},
    ] + make_pairs({"Swahili": 20})
    train, evaluation = split_by_base_stem(pairs)
    sides = [any(p["row_id"] == "R1" for p in side) for side in (train, evaluation)]
    assert sides.count(True) == 1


def test_split_never_leaks_one_question_across_languages():
    # The regression this whole function exists for: 265 of 566 real questions appear in
    # more than one language, and a row_id-level split put 54% of eval questions into
    # training under a different language. Same stems, two languages, must not straddle.
    pairs = make_pairs({"Swahili": 40, "Hausa": 40}, stem_prefix="Q")
    train, evaluation = split_by_base_stem(pairs)
    assert {p["base_stem"] for p in train} & {p["base_stem"] for p in evaluation} == set()


def test_split_still_serves_both_languages_when_every_question_is_shared():
    # Grouping by stem must not collapse into "one language gets everything".
    pairs = make_pairs({"Swahili": 40, "Hausa": 40}, stem_prefix="Q")
    _, evaluation = split_by_base_stem(pairs)
    assert {p["language"] for p in evaluation} == {"Swahili", "Hausa"}


def test_split_gives_every_language_eval_examples():
    # A global shuffle could starve Nyanja (19 pairs against Swahili's 212) entirely.
    _, evaluation = split_by_base_stem(make_pairs({"Swahili": 212, "Nyanja": 19}))
    assert {p["language"] for p in evaluation} == {"Swahili", "Nyanja"}


def test_split_is_deterministic_for_a_given_seed():
    pairs = make_pairs({"Swahili": 50, "Hausa": 30})
    assert split_by_base_stem(pairs, seed=42) == split_by_base_stem(pairs, seed=42)
    assert split_by_base_stem(pairs, seed=42) != split_by_base_stem(pairs, seed=7)


@pytest.mark.parametrize("fraction", [0.1, 0.2, 0.5])
def test_split_respects_the_requested_eval_fraction(fraction):
    pairs = make_pairs({"Swahili": 100, "Hausa": 100})
    _, evaluation = split_by_base_stem(pairs, eval_fraction=fraction)
    assert len(evaluation) == pytest.approx(200 * fraction, abs=2)
