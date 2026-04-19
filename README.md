# Refusal Suppression

Locate, ablate, and evaluate the refusal direction in instruction-tuned LLMs using activation geometry and directional weight editing.

## Key Findings

Using **Llama-3.1-8B-Instruct** on the [OR-Bench Hard-1K](https://huggingface.co/datasets/bench-llm/OR-Bench) benchmark:

| Metric | Base Model | After Edit (layer 14, attn_out, α=0.5) |
|--------|-----------|---------------------------------------|
| Refusal rate (safe borderline prompts) | ~2.5% | **0%** |
| Refusal rate (unsafe prompts) | ~9% | **0%** |
| Capability retention | 100% | **100%** |

**Primary finding:** Refusal behavior in Llama 3.1 8B is geometrically localized to a single direction in the residual stream at layer 14. Ablating this direction at the attention output projection completely suppresses refusal with no measurable capability damage.

**Secondary finding:** The identified refusal direction does not distinguish between true refusals (unsafe prompts) and false refusals (safe but borderline prompts). Both map to the same low-dimensional subspace, suggesting that RLHF-trained refusal is a single unified mechanism rather than a context-sensitive safety classifier.

## Method

1. **Activation probing** — Collect per-layer residual activations for unsafe (refused) vs. benign (answered) prompts
2. **Direction extraction** — Compute the mean-difference direction separating the two groups
3. **Directional ablation** — Project out the refusal direction from attention-out weight matrices at target layers
4. **Grid search** — Search over layers, module types, and edit strengths on a validation split
5. **Holdout evaluation** — Evaluate the best edit on unseen prompts across safety groups

## Pipeline

All experiments run on Kaggle T4x2 (15 GB VRAM per GPU) using 4-bit quantization.

### 1. Measure activations on the discovery split

```bash
python scripts/measure_activations.py \
	--model-id meta-llama/Llama-3.1-8B-Instruct \
	--split-path data/processed/splits/orbench/discovery.jsonl \
	--output artifacts/activations/llama31_8b_bf16_orbench.json \
	--group unsafe_true_refusal --group benign_easy \
	--capture-default-modules --prompt-limit 100
```

### 2. Compute the refusal direction

```bash
python scripts/compute_directions.py \
	--activations artifacts/activations/llama31_8b_bf16_orbench.json \
	--source-group-a unsafe_true_refusal \
	--source-group-b benign_easy \
	--output artifacts/directions/llama31_8b_bf16_orbench_unsafe_vs_easy.json
```

### 3. Search edit candidates on the selection split

```bash
python scripts/search_edits.py \
	--model-id meta-llama/Llama-3.1-8B-Instruct \
	--direction-artifact artifacts/directions/llama31_8b_bf16_orbench_unsafe_vs_easy.json \
	--selection-split data/processed/splits/orbench/selection.jsonl \
	--output artifacts/edits/llama31_8b_bf16_orbench_search.json \
	--top-k-layers 3 --strength 0.5 --strength 1.0 \
	--module-type attn_out --module-type mlp_down \
	--load-in-4bit
```

### 4. Evaluate the best edit on holdout

```bash
python scripts/run_eval.py \
	--model-id meta-llama/Llama-3.1-8B-Instruct \
	--prompts data/processed/splits/orbench/holdout.jsonl \
	--direction-artifact artifacts/directions/llama31_8b_bf16_orbench_unsafe_vs_easy.json \
	--candidate-json artifacts/edits/llama31_8b_bf16_orbench_search.json \
	--output artifacts/eval/llama31_8b_bf16_orbench_holdout.json \
	--load-in-4bit --prompt-limit 200
```

## Data

**Primary benchmark:** OR-Bench Hard-1K (1,319 safe borderline + 655 unsafe prompts), converted via `scripts/convert_orbench.py` and split into discovery / selection / holdout sets under `data/processed/splits/orbench/`.

**Pilot benchmarks** (used during development):
- [data/processed/prompts/sample_prompts.jsonl](data/processed/prompts/sample_prompts.jsonl) — smoke-test data
- [data/processed/prompts/pilot_prompts.jsonl](data/processed/prompts/pilot_prompts.jsonl) — 64-prompt pilot
- [data/processed/prompts/pilot_prompts_gemini.jsonl](data/processed/prompts/pilot_prompts_gemini.jsonl) — 128-prompt Gemini-augmented pilot

## Notebooks

The notebooks under [notebooks/](notebooks/) run on Kaggle T4x2 or Colab with auto-detection:

| Notebook | Purpose |
|----------|---------|
| `00_data_audit.ipynb` | Inspect prompt distributions and split balance |
| `10_activation_geometry.ipynb` | Visualize per-layer activation separability |
| `20_edit_search.ipynb` | Run grid search over edit candidates |
| `30_eval_and_error_analysis.ipynb` | Holdout evaluation with topic-level error analysis |

Each notebook clones the repo, installs dependencies, and reads HF tokens from Kaggle/Colab secrets automatically.

## Project Structure

```
src/frs/           Core library
  data/            Prompt loading, augmentation, splits
  editing/         Directional ablation, search, projection
  evaluation/      Refusal detection, metrics, reports
  models/          Model loading, generation, hooks
  training/        QLoRA repair (optional)
scripts/           CLI entry points for each pipeline stage
configs/           YAML configs for data, models, edits, training
artifacts/         Generated outputs (activations, directions, edits, eval reports)
notebooks/         Kaggle/Colab notebooks for each stage
```
