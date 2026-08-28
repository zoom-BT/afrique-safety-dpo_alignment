# Data — provenance and attribution

## UbuntuGuard (`Ubuntu_guard_test_*.jsonl`)

Redistributed here so a single `git clone` gives a runnable pipeline, which is what makes
the evaluation reproducible in one command rather than in one command plus a manual
download step. 30 MB for the three files.

**Source:** Abdullahi, T., Mgonzo, M., Oduwole, M., Okewunmi, P., Owodunni, A., Singh, R.,
& Eickhoff, C. (2026). *UbuntuGuard: A culturally-grounded policy benchmark for equitable
AI safety in African languages.* arXiv:2601.12696.

**Upstream repository:** https://github.com/hemhemoh/UbuntuGuard — files taken verbatim
from commit `9213a82` ("Add code and data for model evaluation", 2026-04-15), unmodified.

**License:** CC BY 4.0, as stated in the paper. ⚠️ The upstream repository carries no
LICENSE file, so this rests on the paper's statement alone; confirmation was requested from
the corresponding author and is still outstanding. Attribution is given above as CC BY 4.0
requires. Re-check before this data travels anywhere beyond this private repository.

### What is and is not here

Only the **test** splits exist upstream. The paper's Table 3 reports per-language train and
test sizes, but no training file has ever been committed — verified across the repository's
full git history, not just its current file listing. Training data is therefore carved out
of these test files; see the vault's `Week5_Deviations_From_Proposal.md` (D1) and
`Week5_Dataset_Description_Sheet.md`.

| File | Rows | Unique `row_id` | Content |
| :---- | ---: | ---: | :---- |
| `Ubuntu_guard_test_all_english_only.jsonl` | 2,449 | 903 | English dialogue, English policy |
| `Ubuntu_guard_test_crosslingual.jsonl` | 2,307 | 851 | African dialogue, **English** policy |
| `Ubuntu_guard_test_translated.jsonl` | 2,307 | 851 | African dialogue, **localised** policy |

All eleven per-language counts match the paper's Table 3 exactly.

`crosslingual` and `translated` differ in the `policy` field **only** — 2,307 of 2,307 rows
differ there, none differ in `transcript`. They are one set of dialogues under two policy
languages, so their pools do not add up.

### Two parsing traps

Both are handled in `src/data.py`, and both will bite anyone reading these files fresh:

- Turns are marked `User:` / `Agent:` — not `Assistant:` — and `Agent:` is usually indented
  by a space rather than starting at column 0.
- `metadata` is a *Python* dict repr with single quotes, not JSON. `json.loads` raises on
  it; use `ast.literal_eval`.

## Datasets not stored here

Downloaded at runtime via `load_dataset()`, since they are small and their licences vary:

| Dataset | License | Role |
| :---- | :---- | :---- |
| `masakhane/uhura-truthfulqa` | MIT | Honest-axis training pairs and evaluation |
| `masakhane/afrimgsm` (IrokoBench) | Apache-2.0 | utility control |
| AfriHate | Apache-2.0 | Harmless-axis evaluation (macro F1) |
| `McGill-NLP/tukabench` | **CC-BY-NC 4.0** | Harmless-axis evaluation — non-commercial only |
| HealthBench-Africa Extension | unconfirmed | over-refusal evaluation |

On Kaggle these need internet enabled in the notebook settings and an `HF_TOKEN` in Kaggle
Secrets, otherwise Hub requests are rate-limited as unauthenticated.
