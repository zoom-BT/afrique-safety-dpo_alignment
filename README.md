# afrique-safety-dpo

Code for the Week 5-8 research topic: *Translated Safety Alignment vs. Native — Direct
Preference Optimization on African Multi-Lingual Foundation Models*.

The research proposal, daily logs, and reading notes for this internship live in a
separate private repository (an Obsidian vault); this repository holds only the
training/evaluation code for this specific topic, kept independent from the earlier
weeks' SFT/LoRA/QLoRA/ORPO experiments.

## Status

Repository scaffolding only. `src/train.py` carries over the QLoRA/DPO/PEFT helpers
already tested in earlier weeks (quantization config, warmup-step conversion, checkpoint
cleanup, training-curve logging). Not yet implemented: formatting UbuntuGuard's PASS/FAIL
training-split pairs into the Native-DPO dataset, and the NLLB-translated counterfactual
for Translated-DPO — see the `TODO` in `config.yaml`'s `dpo` section.

## Setup

```bash
pip install -r requirements.txt
pytest
```

## Structure

- `src/train.py` — DPO training entry point (`run_dpo`), PEFT/QLoRA config builders.
- `src/utils.py` — seeding, device selection, GPU memory logging.
- `src/evaluate.py` — generation helpers for qualitative checks and downstream RR/Over-RR scoring.
- `config.yaml` — model, training, and DPO hyperparameters.
- `notebooks/` — Kaggle/Colab notebooks (not yet added).

## Privacy

Private repository. No public release without the internship supervisor's approval.
