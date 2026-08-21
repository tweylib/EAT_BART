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
- the same seed, sequence lengths, optimizer settings, batch sizes, gradient
  accumulation, epoch count, and generation parameters;
- the same automatic metrics and two-judge aggregation protocol.

The architecture is the intended independent variable: standard BART versus
emotion-aware BART.

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

Train the baseline for at most 50 epochs. Training stops after two consecutive
epochs without an improvement in validation loss, and the trainer reloads the
checkpoint with the lowest validation loss before saving the final model:

```bash
python scripts/train.py --config configs/kaggle_baseline_5epoch.yaml
```

Evaluate the full test split and score automatic metrics:

```bash
python scripts/evaluate.py --config configs/kaggle_baseline_5epoch_experiment_evaluate.yaml
python scripts/score_generations.py --config configs/kaggle_baseline_5epoch_experiment_score.yaml
```

Run both Groq judges and aggregate completion-weighted scores:

```bash
python scripts/judge_generations.py --config configs/kaggle_baseline_5epoch_experiment_judge_groq.yaml
python scripts/judge_generations.py --config configs/kaggle_baseline_5epoch_experiment_judge_groq_gpt_oss.yaml
python scripts/aggregate_judges.py --config configs/kaggle_baseline_5epoch_experiment_judge_groq_2judge_aggregate.yaml
```

Targets and generated responses are capped at 512 BART tokens. Automatic BLEU
is computed with SacreBLEU and reported on its standard 0-100 scale.
Training and evaluation losses are logged at every epoch and written together
to `epoch_losses.csv` in the model output directory.
