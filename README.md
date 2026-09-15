# EAT-BART

Encoder-side Emotion-Aware Transformer using `facebook/bart-base` for mental-health response generation.

This branch contains the encoder-only EAT ablation. NRC Emotion Intensity Lexicon
features are injected into BART encoder self-attention. Decoder self-attention and
cross-attention remain standard BART.

## Project Shape

- `configs/`: local and Kaggle experiment settings.
- `src/eat_bart/data/`: datasets, token emotion features, and NRC lexicon loading.
- `src/eat_bart/modeling/`: EAT attention modules and BART patching.
- `src/eat_bart/training/`: training, evaluation, and metrics helpers.
- `src/eat_bart/utils/`: config, seed, and device utilities.
- `scripts/`: command-line entry points.
- `tests/`: focused tests for shapes, masking, lexicon features, and BART patching.

## Current Kaggle Protocol

For the controlled baseline-to-EAT comparison, the EAT configuration loads the
complete baseline checkpoint from
`/kaggle/input/datasets/cheikhtidjanitweylib/baseline-30-eps-new-model/models/bart_baseline_comparable`.
The EAT preflight verifies its `run_manifest.json` before training.

```bash
python scripts/check_comparability.py --config configs/kaggle_encoder_eat_comparable.yaml
python scripts/train.py --config configs/kaggle_encoder_eat_comparable.yaml
python scripts/evaluate.py --config configs/kaggle_encoder_eat_comparable_evaluate.yaml
python scripts/score_generations.py --config configs/kaggle_encoder_eat_comparable_score.yaml
python scripts/judge_generations.py --config configs/kaggle_encoder_eat_comparable_judge_gpt_oss.yaml
python scripts/judge_generations.py --config configs/kaggle_encoder_eat_comparable_judge_qwen.yaml
python scripts/aggregate_judges.py --config configs/kaggle_encoder_eat_comparable_judge_aggregate.yaml
```

To test whether the original `3e-4` EAT learning rate was too large, run the
controlled `1e-4` diagnostic below. It keeps alpha at `0.10` and inherits the
same seed, baseline checkpoint, effective batch size, 40-epoch ceiling, and
early-stopping patience. Its model and reports use separate paths, so the
completed `3e-4` run is not overwritten.
The contextual feature cache is stored under `/kaggle/working/cache`; the first
run builds it there because Kaggle's `/kaggle/input` datasets are read-only.

```bash
python scripts/check_comparability.py --config configs/kaggle_encoder_eat_comparable_a010_lr1e4.yaml
python scripts/train.py --config configs/kaggle_encoder_eat_comparable_a010_lr1e4.yaml
python scripts/evaluate.py --config configs/kaggle_encoder_eat_comparable_a010_lr1e4_evaluate.yaml
python scripts/score_generations.py --config configs/kaggle_encoder_eat_comparable_a010_lr1e4_score.yaml
python scripts/judge_generations.py --config configs/kaggle_encoder_eat_comparable_a010_lr1e4_judge_gpt_oss.yaml
python scripts/judge_generations.py --config configs/kaggle_encoder_eat_comparable_a010_lr1e4_judge_qwen.yaml
python scripts/aggregate_judges.py --config configs/kaggle_encoder_eat_comparable_a010_lr1e4_judge_aggregate.yaml
```

Run evaluation and scoring only after training completes. Compare the best
validation loss with `0.7767994404`; improvement beyond ordinary rerun noise
supports the learning-rate hypothesis, while another flat curve points toward
limited leverage from the current fixed-alpha EAT parameterization.

The Qwen judge runs with reasoning disabled and JSON mode enabled. This keeps
its short scoring response from spending the output budget inside an incomplete
`<think>` block. Rerun the complete 100-example Qwen evaluation after changing
these settings; do not combine partial results produced by the old settings.

Run the final controlled alpha experiment below. It inherits the complete
`alpha=0.10`, `LR=1e-4` protocol and changes only alpha to `0.05` plus its
isolated artifact names.

```bash
python scripts/check_comparability.py --config configs/kaggle_encoder_eat_comparable_a005_lr1e4.yaml
python scripts/train.py --config configs/kaggle_encoder_eat_comparable_a005_lr1e4.yaml
python scripts/evaluate.py --config configs/kaggle_encoder_eat_comparable_a005_lr1e4_evaluate.yaml
python scripts/score_generations.py --config configs/kaggle_encoder_eat_comparable_a005_lr1e4_score.yaml
python scripts/judge_generations.py --config configs/kaggle_encoder_eat_comparable_a005_lr1e4_judge_gpt_oss.yaml
python scripts/judge_generations.py --config configs/kaggle_encoder_eat_comparable_a005_lr1e4_judge_qwen.yaml
python scripts/aggregate_judges.py --config configs/kaggle_encoder_eat_comparable_a005_lr1e4_judge_aggregate.yaml
```

Evaluate the trained EAT checkpoint with its emotion branch disabled, without
performing any further training. Evaluation loads the uploaded checkpoint at
`/kaggle/input/datasets/cheikhmohamedahid/eat-encoder/models/encoder_eat_comparable`
and the uploaded contextual cache under the same dataset. Run these commands
from the repository's `EAT_BART` directory:

```bash
python scripts/evaluate.py --config configs/kaggle_encoder_eat_comparable_alpha0_evaluate.yaml
python scripts/score_generations.py --config configs/kaggle_encoder_eat_comparable_alpha0_score.yaml
```

The manifest check covers the dataset hash, split, tokenizer behavior, source
and target lengths, seed, precision, and batch/accumulation settings. The
baseline and EAT stages use the same effective global batch; only their epoch
ceilings, learning rates, and trainable parameters intentionally differ.

Train the cleaned 5-epoch encoder-EAT model:

```bash
python scripts/train.py --config configs/kaggle_encoder_only_5epoch.yaml
```

Evaluate on the full test split and score automatic metrics:

```bash
python scripts/evaluate.py --config configs/kaggle_encoder_only_5epoch_experiment_evaluate.yaml
python scripts/score_generations.py --config configs/kaggle_encoder_only_5epoch_experiment_score.yaml
```

Run the two Groq LLM judges and aggregate with completion-weighted scores:

```bash
python scripts/judge_generations.py --config configs/kaggle_encoder_only_5epoch_experiment_judge_groq.yaml
python scripts/judge_generations.py --config configs/kaggle_encoder_only_5epoch_experiment_judge_groq_gpt_oss.yaml
python scripts/aggregate_judges.py --config configs/kaggle_encoder_only_5epoch_experiment_judge_groq_2judge_aggregate.yaml
```

## Attention Contract

Base attention scores:

```text
A = QK^T / sqrt(d_k)
```

Main EAT formula:

```text
A_eat = A + alpha_h * S_h
```

Ablation formula:

```text
A_eat = A * (I + alpha_h * S_h)
```

Padding and causal masks are applied last with `masked_fill`.
