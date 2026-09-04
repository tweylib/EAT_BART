# Standard BART Baseline

This branch is the baseline ablation for the EAT-BART project. It fine-tunes an
unmodified `BartForConditionalGeneration` loaded from `facebook/bart-base` for
mental-health response generation.

No EAT attention patches, emotion interaction parameters, NRC lexicon features,
or emotion tensors are used. Encoder self-attention, decoder self-attention, and
cross-attention are all the stock Hugging Face BART implementation.

## Comparison Contract

The baseline keeps the experimental protocol inherited from the encoder-EAT
comparison branch:

- the same dataset columns and deterministic train/validation/test split;
- the same seed, sequence lengths, microbatch size, gradient accumulation, and
  generation parameters;
- the same automatic metrics and GPT-OSS judging protocol.

The epoch ceilings, learning rates, and trainable parameter sets intentionally
differ between baseline fitting and the subsequent frozen-BART EAT stage.

## Project Shape

- `configs/`: local and Kaggle baseline experiment settings.
- `src/eat_bart/data/`: dataset, tokenizer, and standard BART collation.
- `src/eat_bart/training/`: training, generation, metrics, and judging helpers.
- `src/eat_bart/utils/`: configuration, seed, and device utilities.
- `scripts/`: command-line entry points.
- `tests/`: focused baseline pipeline and evaluation tests.

The `eat_bart` Python namespace remains unchanged so shared experiment tooling
can import the worktrees consistently; it does not imply EAT is enabled here.

## Kaggle Protocol

Train the baseline for at most 30 epochs. Training stops after three consecutive
epochs without an improvement in validation loss, and the trainer reloads the
checkpoint with the lowest validation loss before saving the final model:

```bash
python scripts/train.py --config configs/kaggle_baseline_comparable.yaml
```

Evaluate the full test split and score automatic metrics:

```bash
python scripts/evaluate.py --config configs/kaggle_baseline_comparable_evaluate.yaml
python scripts/score_generations.py --config configs/kaggle_baseline_comparable_score.yaml
```

Run the same GPT-OSS and Qwen judges used for EAT on the first 100 test examples:

```bash
python scripts/judge_generations.py --config configs/kaggle_baseline_comparable_judge_gpt_oss.yaml
python scripts/judge_generations.py --config configs/kaggle_baseline_comparable_judge_qwen.yaml
python scripts/aggregate_judges.py --config configs/kaggle_baseline_comparable_judge_aggregate.yaml
```

The training output includes `run_manifest.json`. Upload the complete
`/kaggle/working/models/bart_baseline_comparable` directory as a Kaggle input;
the EAT stage will refuse to train if its protocol or dataset does not match.

Targets and generated responses are capped at 512 BART tokens. Automatic BLEU
is computed with SacreBLEU and reported on its standard 0-100 scale.
Training and evaluation losses are logged at every epoch and written together
to `epoch_losses.csv` in the model output directory.
