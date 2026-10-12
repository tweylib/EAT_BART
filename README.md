# Decoder EAT-BART

This branch implements the decoder-self-attention placement ablation for
mental-health response generation with `facebook/bart-base`. Encoder
self-attention and decoder cross-attention remain native BART. The generator
and GoEmotions model are frozen; only decoder EAT W1/W2 tensors train.

## Decoder emotion features

The frozen `SamLowe/roberta-base-go_emotions` model processes decoder tokens
with a triangular attention mask. A feature at position `t` can therefore use
only the response prefix through `t`.

- Training builds the causal features once and caches them by response text.
- The collator aligns those cached features to shifted BART decoder inputs.
- Generation recomputes features from the current generated prefix after beam
  expansion or reordering. It never uses the reference response.
- Every generated BART prefix must round-trip through canonical text exactly;
  the run fails instead of guessing if alignment changes.

The canonical experiment config is `configs/kaggle_decoder_eat.yaml`. It keeps
the comparison setup fixed: the same baseline checkpoint, split and seed,
tokenization limits, effective batch size, probability-mixture formula,
tied-orthogonal initialization, `d_s=64`, `1e-4` learning rate, and alpha
warmup from `0.005` to `0.05` over five epochs.

Padding and causal masks are applied last with `masked_fill`, after the
standard and emotion attention contributions are combined.

## Kaggle smoke test

Attach these two Kaggle datasets before running:

- `cheikhmohamedahid/eat-data`
- `cheikhtidjanitweylib/baseline-30-eps-new-model`

Use a two-GPU accelerator because the comparison guard expects two CUDA
devices. Internet must be enabled the first time unless the Hugging Face models
are already available in the notebook cache.

Kaggle may update its preinstalled PyTorch between the baseline and EAT runs.
The decoder config permits that version drift with a visible warning while
still requiring exact dataset, protocol, Transformers, Accelerate, and
tokenizers matches. Both PyTorch versions remain recorded in the manifests.

```bash
python scripts/check_comparability.py --config configs/kaggle_decoder_eat.yaml
python scripts/train.py --config configs/kaggle_decoder_eat_smoke.yaml
python scripts/evaluate.py --config configs/kaggle_decoder_eat_smoke_evaluate_greedy.yaml
python scripts/evaluate.py --config configs/kaggle_decoder_eat_smoke_evaluate_beam.yaml
```

The smoke run uses 64 examples and five optimizer steps. It fails if decoder
W1/W2 do not receive a finite, nonzero aggregate update, and it saves a small
checkpoint so both greedy and four-beam generated-token paths can be exercised.

## Full comparison

Run the full experiment only after the four smoke commands succeed:

```bash
python scripts/train.py --config configs/kaggle_decoder_eat.yaml
python scripts/evaluate.py --config configs/kaggle_decoder_eat_evaluate.yaml
python scripts/score_generations.py --config configs/kaggle_decoder_eat_score.yaml
```

Evaluation uses four beams to match the baseline protocol. Beam search is not
part of training; it keeps four candidate generated prefixes during inference
and chooses the best sequence under the configured length and repetition
penalties.

The alpha-zero, causal-shuffled, and zero-feature controls are separate configs:

```bash
python scripts/evaluate.py --config configs/kaggle_decoder_eat_alpha0_evaluate.yaml
python scripts/evaluate.py --config configs/kaggle_decoder_eat_shuffled_evaluate.yaml
python scripts/evaluate.py --config configs/kaggle_decoder_eat_zero_evaluate.yaml
```

The shuffled condition is prefix-stable and causal: each content position gets
a deterministic feature drawn from its strict history. It does not permute the
complete response or expose future tokens.

## Repository layout

- `configs/`: decoder training, smoke, evaluation, scoring, and judge settings.
- `src/eat_bart/data/`: datasets and causal decoder emotion features.
- `src/eat_bart/modeling/`: EAT attention and BART integration.
- `src/eat_bart/training/`: training, evaluation, diagnostics, and metrics.
- `scripts/`: command-line entry points.
- `tests/`: unit and integration coverage.

The original placement experiment remains available in Git history and on the
`ablation/encoder-eat` branch; its old run configs are intentionally absent from
this decoder branch.
