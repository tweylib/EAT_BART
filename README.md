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
