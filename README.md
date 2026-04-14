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

## Starter Data

The repo includes a small grouped starter prompt set at [data/processed/prompts/sample_prompts.jsonl](data/processed/prompts/sample_prompts.jsonl) and an exact family-preserving split config at [configs/data/prompt_sets_small.yaml](configs/data/prompt_sets_small.yaml).

Generate discovery, selection, and holdout manifests with:

```bash
python scripts/make_splits.py \
	--input data/processed/prompts/sample_prompts.jsonl \
	--output-dir data/processed/splits/sample_small \
	--config configs/data/prompt_sets_small.yaml \
	--seed 7
```

## Pipeline

1. Measure activations on the discovery split.

```bash
python scripts/measure_activations.py \
	--model-id Qwen/Qwen3-4B \
	--split-path data/processed/splits/sample_small/discovery.jsonl \
	--output artifacts/activations/sample_qwen3_discovery.json \
	--capture-default-modules
```

2. Compute candidate directions.

```bash
python scripts/compute_directions.py \
	--activations artifacts/activations/sample_qwen3_discovery.json \
	--source-group-a benign_borderline \
	--source-group-b benign_easy \
	--output artifacts/directions/sample_qwen3_borderline_vs_easy.json
```

3. Search model edits on the selection split.

```bash
python scripts/search_edits.py \
	--model-id Qwen/Qwen3-4B \
	--direction-artifact artifacts/directions/sample_qwen3_borderline_vs_easy.json \
	--selection-split data/processed/splits/sample_small/selection.jsonl \
	--output artifacts/edits/sample_qwen3_search.json
```

4. Evaluate the best edit on holdout.

```bash
python scripts/run_eval.py \
	--model-id Qwen/Qwen3-4B \
	--prompts data/processed/splits/sample_small/holdout.jsonl \
	--direction-artifact artifacts/directions/sample_qwen3_borderline_vs_easy.json \
	--candidate-json artifacts/edits/sample_qwen3_search.json \
	--output artifacts/eval/sample_qwen3_holdout.json
```

5. Optionally run repair training.

```bash
python scripts/train_qlora_repair.py \
	--model-id Qwen/Qwen3-4B \
	--prompts data/processed/splits/sample_small/selection.jsonl \
	--output-dir artifacts/repair/sample_qwen3 \
	--output artifacts/repair/sample_qwen3.json
```

## Notebooks

The notebooks under [notebooks](notebooks) are intended for Kaggle-backed execution. Because this repo does not currently have a configured git remote, the notebook bootstrap cells expect an environment variable named `FRS_REPO_URL` that points to the cloneable URL for this repository.

Once that variable is set in the Kaggle environment, each notebook will:

- clone or refresh the repo under `/kaggle/working/false-refusal-suppression`
- install the project in editable mode
- add `src/` to `sys.path`
- run the corresponding stage of the experiment from inside the cloned repo
