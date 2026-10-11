# Decoder EAT design

## Comparison contract

This branch changes only EAT placement and the causal feature plumbing required
by autoregressive decoding. It retains the completed comparison's baseline
checkpoint, dataset split, seed, tokenization limits, effective batch size,
frozen-backbone policy, probability-mixture attention, tied-orthogonal W
initialization, hidden size, optimizer learning rate, and alpha schedule.

Placement is explicit:

- BART encoder self-attention: native.
- BART decoder self-attention: EAT.
- BART decoder cross-attention: native.

## Why decoder handling differs

Source tokens are fully known before generation, but response tokens appear one
at a time. Feeding a complete reference response to the emotion model during
generation would leak the future and make training and inference definitions
different.

The decoder implementation instead uses a frozen GoEmotions RoBERTa with a
triangular self-attention mask. During teacher-forced training the resulting
features are cached against shifted decoder inputs. During free generation they
are recomputed from the live prefix. Beam-expanded and reordered prefixes are
handled independently, so each candidate receives features for its own history.

## Cache contract

The training cache is an efficiency optimization, not a different signal. Its
fingerprint includes response texts, emotion model, maximum target length,
tokenizer prefix-space behavior, decoder start token, and the causal-definition
version. A mismatched or stale cache is rejected.

Each stored row contains the frozen contextual states for one teacher-forced
decoder sequence. The collator pads and batches those rows. Since the emotion
model is frozen and the training response is fixed, recomputing those states on
every epoch would produce the same values and waste GPU time.

The cache is not used for generated responses because their prefixes are not
known in advance.

## Required preflight

Before a full run:

1. Validate the baseline manifest and two-GPU runtime.
2. Complete five smoke optimizer steps and observe a finite nonzero W1/W2 update.
3. Load the saved smoke checkpoint for greedy generation.
4. Load it again for four-beam generation.
5. Start the full experiment only if all four commands exit successfully.

Canonical commands and Kaggle paths are documented in the repository README.
