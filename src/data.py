"""Build DPO preference pairs from the UbuntuGuard benchmark's released test data.

UbuntuGuard ships no training split (see the vault's Week5_Deviations_From_Proposal.md,
D1), so the training data here is carved out of the released test files. Every function
below is deliberately pure and file-format-agnostic where possible, so the split logic can
be tested without the 10 MB JSONL files present.

Layout of a source row (all values are strings):

    policy, transcript, label ("PASS"/"FAIL"), metadata, row_id, base_id,
    country_code, country, language, theme, topic, domain, sensitive_characteristic

Two traps in this data, both handled here:

- Turns are marked `User:` / `Agent:` (not `Assistant:`), and the `Agent:` marker is
  usually indented by a space rather than sitting at column 0.
- `metadata` is a *Python* dict repr with single quotes, not JSON — `json.loads` raises on
  it. Parsed with `ast.literal_eval` in `parse_metadata` below.
"""

import ast
import json
import re
from collections import defaultdict
from pathlib import Path

# Matches a turn marker at the start of the transcript or of any line, tolerating the
# leading whitespace UbuntuGuard puts before `Agent:`.
TURN_MARKER = re.compile(r"(?:\A|\n)[ \t]*(User|Agent):[ \t]*")

ROLE_BY_MARKER = {"User": "user", "Agent": "assistant"}


def base_stem(base_id: str) -> str:
    """Strip the generator suffix from a `base_id` to get the underlying prompt's identity.

    `base_id` looks like `GHA1002_llama` — a source-question id plus the model that produced
    the dialogue. The stem `GHA1002` is what identifies the *question*, and the same stem
    recurs across languages (265 of 566 stems appear in more than one) and across the
    English-only file (all 566 African stems are also in it). Splitting on anything finer
    than the stem therefore leaks a question from train into eval in a different language.
    """
    return base_id.rsplit("_", 1)[0]


def parse_metadata(raw: str) -> dict:
    """Parse UbuntuGuard's `metadata` field, which is a Python dict repr, not JSON.

    `ast.literal_eval` rather than `eval` so a malformed value can only raise, never
    execute anything.
    """
    return ast.literal_eval(raw)


def parse_transcript(transcript: str) -> list[dict]:
    """Split a raw transcript string into `[{"role": ..., "content": ...}, ...]`.

    Roles are normalised to the ChatML names (`user`/`assistant`) that `trl` expects.
    Returns `[]` for a transcript with no recognisable turn markers rather than raising,
    so a single malformed row can be dropped by the caller instead of failing the run.
    """
    markers = list(TURN_MARKER.finditer(transcript))
    if not markers:
        return []

    turns = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(transcript)
        content = transcript[marker.end() : end].strip()
        turns.append({"role": ROLE_BY_MARKER[marker.group(1)], "content": content})
    return turns


def _is_well_formed(turns: list[dict]) -> bool:
    """A usable dialogue alternates user/assistant, starts on `user`, ends on `assistant`.

    Rules out the handful of truncated transcripts (e.g. two user turns and one agent turn)
    that would otherwise produce a prompt with no response to contrast against.
    """
    if len(turns) < 2 or turns[0]["role"] != "user" or turns[-1]["role"] != "assistant":
        return False
    expected = ["user", "assistant"] * (len(turns) // 2)
    return len(turns) % 2 == 0 and [t["role"] for t in turns] == expected


def build_pair(passing: dict, failing: dict, include_policy: bool = True) -> dict | None:
    """Build one DPO preference pair from a PASS row and a FAIL row sharing a `row_id`.

    The two transcripts share a leading prefix and then diverge. Everything up to the first
    differing turn becomes the DPO `prompt`; the two divergent assistant responses become
    `chosen` and `rejected`. Whatever follows in either transcript is discarded.

    In practice the divergence is at turn 1 — the *first* assistant response — for 839 of
    the 843 pairable dialogues, which is where the safety decision is actually made. That
    also makes the truncation the right call rather than a loss: the later turns of a FAIL
    transcript are conditioned on an already-unsafe response, so keeping them would blur
    "produced an unsafe answer" together with "kept going down that path".

    Returns `None` when the pair is unusable — malformed turns, no divergence at all, or a
    divergence on a *user* turn, which would mean the two sides answer different questions
    and the comparison no longer isolates the model's own behaviour.

    `include_policy` prepends the row's safety policy as a system message. On by default
    because UbuntuGuard is a *policy-based* benchmark: PASS and FAIL are defined relative
    to that policy, so without it the preference signal is only weakly grounded.
    """
    chosen_turns = parse_transcript(passing["transcript"])
    rejected_turns = parse_transcript(failing["transcript"])

    if not _is_well_formed(chosen_turns) or not _is_well_formed(rejected_turns):
        return None

    divergence = next(
        (i for i, (c, r) in enumerate(zip(chosen_turns, rejected_turns)) if c != r), None
    )
    if divergence is None:
        return None
    if chosen_turns[divergence]["role"] != "assistant":
        return None

    prompt = chosen_turns[:divergence]
    if include_policy:
        prompt = [{"role": "system", "content": passing["policy"]}] + prompt

    return {
        "prompt": prompt,
        "chosen": [chosen_turns[divergence]],
        "rejected": [rejected_turns[divergence]],
        "row_id": passing["row_id"],
        "base_stem": base_stem(passing["base_id"]),
        "language": passing["language"],
        "domain": passing["domain"],
        "theme": passing["theme"],
    }


def build_preference_pairs(rows: list[dict], include_policy: bool = True) -> list[dict]:
    """Turn raw UbuntuGuard rows into DPO preference pairs, grouped by `row_id`.

    A `row_id` may carry more than one PASS and more than one FAIL (271 of them carry two
    of each). Those are matched up positionally so that **no response is ever reused across
    two pairs** — reusing one would inflate the apparent dataset size while feeding the same
    text to the loss twice. On the African test files this yields 1,089 usable pairs, against
    the 501 obtained by keeping only `row_id`s with exactly one PASS and one FAIL.

    Rows are sorted by `base_id` before matching so the output is deterministic regardless
    of input order.
    """
    by_row_id: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: {"PASS": [], "FAIL": []}
    )
    for row in rows:
        if row["label"] in by_row_id[row["row_id"]]:
            by_row_id[row["row_id"]][row["label"]].append(row)

    pairs = []
    for row_id in sorted(by_row_id):
        group = by_row_id[row_id]
        passing = sorted(group["PASS"], key=lambda r: r["base_id"])
        failing = sorted(group["FAIL"], key=lambda r: r["base_id"])
        for pass_row, fail_row in zip(passing, failing):
            pair = build_pair(pass_row, fail_row, include_policy=include_policy)
            if pair is not None:
                pairs.append(pair)
    return pairs


def split_by_base_stem(
    pairs: list[dict], eval_fraction: float = 0.2, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Split pairs into train/eval with no underlying question on both sides.

    Grouping at `base_stem` level is what prevents contamination, and it has to be that
    coarse. Two weaker groupings both leak:

    - Splitting per *pair* leaks verbatim: several pairs share one `row_id` and therefore
      an identical prompt.
    - Splitting per `row_id` leaks *across languages*: 265 of the 566 underlying questions
      appear in more than one language, so the same question can land in train as Swahili
      and in eval as Hausa. For a study about cross-lingual safety transfer that is fatal —
      it would measure memorisation and report it as transfer. Measured on this corpus, a
      `row_id`-level split put 85 of 156 eval questions (54%) into training in another
      language.

    This is the Week 5 contract's "investigate duplication and contamination" task,
    enforced in code rather than assumed.

    Languages are filled rarest-first. A stem can carry several languages, so whichever
    language claims it first decides where it lands; going commonest-first would let
    Swahili absorb the shared stems and leave Nyanja (19 pairs) with no eval set at all.
    """
    import random

    pairs_by_stem: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        pairs_by_stem[pair["base_stem"]].append(pair)

    stems_by_language: dict[str, set[str]] = defaultdict(set)
    pair_count_by_language: dict[str, int] = defaultdict(int)
    for pair in pairs:
        stems_by_language[pair["language"]].add(pair["base_stem"])
        pair_count_by_language[pair["language"]] += 1

    rng = random.Random(seed)
    eval_stems: set[str] = set()
    assigned: set[str] = set()

    for language in sorted(pair_count_by_language, key=lambda l: (pair_count_by_language[l], l)):
        quota = round(pair_count_by_language[language] * eval_fraction)
        taken = sum(
            1
            for stem in eval_stems
            for pair in pairs_by_stem[stem]
            if pair["language"] == language
        )
        candidates = sorted(stems_by_language[language] - assigned)
        rng.shuffle(candidates)
        for stem in candidates:
            if taken >= quota:
                break
            eval_stems.add(stem)
            assigned.add(stem)
            taken += sum(1 for p in pairs_by_stem[stem] if p["language"] == language)
        assigned.update(stems_by_language[language])

    train = [p for p in pairs if p["base_stem"] not in eval_stems]
    evaluation = [p for p in pairs if p["base_stem"] in eval_stems]
    return train, evaluation


def load_ubuntuguard_rows(path: str | Path) -> list[dict]:
    """Read one UbuntuGuard `.jsonl` file into a list of row dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_ubuntuguard_dpo_datasets(
    path: str | Path,
    eval_fraction: float = 0.2,
    seed: int = 42,
    include_policy: bool = True,
) -> tuple[list[dict], list[dict]]:
    """End-to-end: UbuntuGuard JSONL file -> contamination-free (train, eval) pair lists.

    Returns plain lists rather than `datasets.Dataset` objects so this stays importable and
    testable without the `datasets` dependency; `train.build_dpo_dataset_from_pairs`
    converts them at the point of use.
    """
    rows = load_ubuntuguard_rows(path)
    pairs = build_preference_pairs(rows, include_policy=include_policy)
    return split_by_base_stem(pairs, eval_fraction=eval_fraction, seed=seed)
