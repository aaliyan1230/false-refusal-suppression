# False Refusal Suppression

Measure, edit, and evaluate false-refusal behavior in small instruct models, with a workflow built for Kaggle-scale experimentation.

## Status

The repo now has a working starter pipeline for:

- grouped prompt normalization and split generation
- activation measurement from per-layer last-token residual states
- direction computation from activation artifacts
- model-space directional edits over attention-out and MLP-down projections
- bounded edit search over layers, spans, strengths, and norm-preserving settings
- live holdout evaluation for base or edited models
- QLoRA repair training via Unsloth when available, with a transformers and PEFT fallback

The remaining GPU-bound step is running the heavy model and notebook workflows in Kaggle.

## Data

The repo includes two prompt inventories:

- [data/processed/prompts/sample_prompts.jsonl](data/processed/prompts/sample_prompts.jsonl): tiny smoke-test data for quick validation
- [data/processed/prompts/pilot_prompts.jsonl](data/processed/prompts/pilot_prompts.jsonl): generated pilot benchmark with 64 prompts and exact grouped splits
- [data/processed/prompts/pilot_prompts_gemini.jsonl](data/processed/prompts/pilot_prompts_gemini.jsonl): Gemini-augmented local benchmark with 128 prompts and exact grouped splits

The pilot benchmark is built from family seeds in [data/raw/prompt_family_seeds.json](data/raw/prompt_family_seeds.json). You can regenerate it deterministically, or add Gemini paraphrases by passing `--use-gemini` to the builder. The script will read `GEMINI_API_KEY` from the environment or from the repo-root `.env` file.

Data augmentation is intended to happen locally, not in Kaggle. The Kaggle notebooks now assume that prompt files and split manifests have already been generated and are present in the cloned repo.

Generate the pilot prompt inventory with:

```bash
python scripts/build_prompt_sets.py \
	--seed-families data/raw/prompt_family_seeds.json \
	--output data/processed/prompts/pilot_prompts.jsonl \
	--examples-per-family 4 \
	--seed 7
```

Then generate discovery, selection, and holdout manifests with:

```bash
python scripts/make_splits.py \
	--input data/processed/prompts/pilot_prompts.jsonl \
	--output-dir data/processed/splits/pilot \
	--config configs/data/prompt_sets_pilot.yaml \
	--seed 7
```

For the Gemini-augmented local benchmark, use:

```bash
python scripts/build_prompt_sets.py \
	--seed-families data/raw/prompt_family_seeds.json \
	--output data/processed/prompts/pilot_prompts_gemini.jsonl \
	--examples-per-family 8 \
	--seed 7 \
	--use-gemini \
	--gemini-model gemini-2.5-flash
```

```bash
python scripts/make_splits.py \
	--input data/processed/prompts/pilot_prompts_gemini.jsonl \
	--output-dir data/processed/splits/pilot_gemini \
	--config configs/data/prompt_sets_gemini_pilot.yaml \
	--seed 7
```

## Pipeline

1. Measure activations on the discovery split.

```bash
python scripts/measure_activations.py \
	--model-id Qwen/Qwen3-4B \
	--split-path data/processed/splits/pilot_gemini/discovery.jsonl \
	--output artifacts/activations/qwen3_gemini_pilot_discovery.json \
	--capture-default-modules
```

2. Compute candidate directions.

```bash
python scripts/compute_directions.py \
	--activations artifacts/activations/qwen3_gemini_pilot_discovery.json \
	--source-group-a benign_borderline \
	--source-group-b benign_easy \
	--output artifacts/directions/qwen3_gemini_pilot_borderline_vs_easy.json
```

3. Search model edits on the selection split.

```bash
python scripts/search_edits.py \
	--model-id Qwen/Qwen3-4B \
	--direction-artifact artifacts/directions/qwen3_gemini_pilot_borderline_vs_easy.json \
	--selection-split data/processed/splits/pilot_gemini/selection.jsonl \
	--output artifacts/edits/qwen3_gemini_pilot_search.json
```

4. Evaluate the best edit on holdout.

```bash
python scripts/run_eval.py \
	--model-id Qwen/Qwen3-4B \
	--prompts data/processed/splits/pilot_gemini/holdout.jsonl \
	--direction-artifact artifacts/directions/qwen3_gemini_pilot_borderline_vs_easy.json \
	--candidate-json artifacts/edits/qwen3_gemini_pilot_search.json \
	--output artifacts/eval/qwen3_gemini_pilot_holdout.json
```

5. Optionally run repair training.

```bash
python scripts/train_qlora_repair.py \
	--model-id Qwen/Qwen3-4B \
	--prompts data/processed/splits/pilot_gemini/selection.jsonl \
	--output-dir artifacts/repair/qwen3_gemini_pilot \
	--output artifacts/repair/qwen3_gemini_pilot.json
```

## Notebooks

The notebooks under [notebooks](notebooks) are intended for Kaggle-backed execution. Because this repo does not currently have a configured git remote, the notebook bootstrap cells expect an environment variable named `FRS_REPO_URL` that points to the cloneable URL for this repository.

Once that variable is set in the Kaggle environment, each notebook will:

- clone or refresh the repo under `/kaggle/working/false-refusal-suppression`
- install the project in editable mode
- add `src/` to `sys.path`
- run the corresponding stage of the experiment from inside the cloned repo
