# EAT-BART Modifications and Experimental Results

**Prepared:** 2026-09-11

**Final encoder-EAT code reviewed:** `ablation/encoder-eat` at `6f6e9c6`

**Comparable baseline code:** `ablation/baseline` at `05c28e5`

## 1. Scope and reference point

This document records the material changes made after the original cleaned
encoder-only EAT-BART implementation, represented by commit `79c9897`, and
summarizes the experimental results obtained during development.

The starting implementation had these principal properties:

- `facebook/bart-base` was loaded directly rather than from a separately
  fine-tuned response-generation baseline.
- Emotion features came from an eight-dimensional NRC lexicon lookup.
- The encoder-side EAT score was formed from head-specific projections:
  \(S_h=(E W^S_{1,h})(E W^S_{2,h})^T/\sqrt{d_s}\).
- Emotion scores were injected before a single softmax, principally through
  the additive form \(\operatorname{softmax}(A+\alpha S)\).
- Alpha was represented as a trainable per-head parameter.
- The encoder-only configuration modified encoder self-attention while leaving
  decoder self-attention and encoder-decoder cross-attention unchanged.
- The original target maximum length was 128, the BART tokenizer used
  `add_prefix_space: true`, and the initial Kaggle microbatch/accumulation
  settings were 2 and 8.

The final controlled experiment is substantially different and should be
treated as a new contextual EAT variant rather than a direct continuation of
the original lexicon experiment.

## 2. Final model formulation

### 2.1 Contextual emotion-informed representations

The NRC lookup was replaced for the new experiment by the frozen sequence
classifier:

```text
SamLowe/roberta-base-go_emotions
```

The 28 classification logits are not used as token features. Instead, the
final hidden layer of the underlying RoBERTa encoder is used, producing
768-dimensional contextual emotion-informed token representations.

The classifier is always placed in evaluation mode, all its parameters have
`requires_grad=False`, and extraction runs under `torch.no_grad()`. It is not
included in an optimizer group. After cache construction, the model is deleted
and its GPU cache is released.

The original NRC lexicon pathway remains available through the configurable
`emotion_feature_source` switch; it was not deleted or overwritten.

### 2.2 BART/RoBERTa token alignment

The implementation does not assume token index equality. Both fast tokenizers
produce character-offset mappings from the same raw question text. For every
valid non-special BART token, the pipeline:

1. checks whether the complete BART and RoBERTa offset tensors are identical
   and uses an identity fast path when they are;
2. otherwise finds every RoBERTa token with positive character-span overlap;
3. mean-pools all overlapping RoBERTa hidden states onto that BART span.

The resulting tensor has exactly one row per BART position:

\[
E\in\mathbb{R}^{B\times L_{BART}\times768}.
\]

Special-token and padding rows are zero. Alignment cache construction aborts
if any valid BART token cannot be aligned, rather than silently inserting an
incorrect representation. Cache fingerprints cover the full text sequence,
model name, maximum length, tokenizer prefix-space behavior, and cache schema.

The successfully constructed full cache therefore had zero unhandled valid
token alignment failures. The later EAT signal diagnostic reported 95.66%
nonzero contextual-feature coverage when special tokens were included among
valid BART sequence positions.

### 2.3 Removal of the additional W_E projection

An intermediate design introduced a trainable
\(W_E\in\mathbb{R}^{768\times128}\) projection. At the user's request, this
matrix was removed. The final formulation feeds the aligned 768-dimensional
RoBERTa states directly to the existing pairwise projections:

\[
Z_{1,h}=E W^S_{1,h},\qquad
Z_{2,h}=E W^S_{2,h},\qquad
S_h=Z_{1,h}Z_{2,h}^T/\sqrt{32}.
\]

For each of BART-base's six encoder layers:

- \(W^S_1\) has shape `[12, 768, 32]`;
- \(W^S_2\) has shape `[12, 768, 32]`;
- the matrices are layer-specific and head-specific, not shared globally.

This leaves 3,538,944 trainable parameters across the six encoder layers. No
`W_E` parameter or optimizer group exists in the final model.

### 2.4 Probability-mixture attention

The additive-logit formulation was replaced with two independently normalized
distributions:

\[
A_h=Q_hK_h^T/\sqrt{d_k},\qquad
P_{A,h}=\operatorname{softmax}(A_h+M),
\]

\[
P_{S,h}=\operatorname{softmax}(S_h+M),
\]

\[
P_h=(1-\alpha)P_{A,h}+\alpha P_{S,h},\qquad O_h=P_hV_h.
\]

There is no additional softmax after the mixture. Both distributions receive
the same key-validity mask, so padding and otherwise forbidden keys receive
zero probability. A Transformers 5 boolean-mask semantic error was found and
corrected: in that backend, `True` denotes an allowed position rather than a
position to suppress.

For the completed experiment, alpha was a fixed scalar buffer with
`alpha=0.10`; it was neither learnable nor part of the optimizer. The code
enforces \(0\leq\alpha\leq1\).

### 2.5 Encoder-only placement

Only the six BART encoder self-attention modules are patched. Decoder
self-attention and encoder-decoder cross-attention remain native BART.
Emotion features are routed through lightweight forward shims while the rest
of Hugging Face BART's forward and generation logic remains intact.

## 3. Training workflow changes

### 3.1 Baseline-first training

The final workflow has two independent stages:

1. Fine-tune a standard BART response-generation baseline and save it.
2. Load that fine-tuned checkpoint, add fresh encoder EAT matrices, freeze the
   entire BART model, and train only \(W^S_1\) and \(W^S_2\).

The frozen GoEmotions classifier is used only to construct cached contextual
states. Thus the final EAT optimizer contains neither BART, RoBERTa, alpha, nor
`W_E` parameters.

### 3.2 Contextual feature cache

Aligned frozen RoBERTa states are cached before the trainable W1/W2 mappings.
The cache uses float16 and was estimated at approximately 1.29 GiB, which is
well within the 30 GiB Kaggle RAM limit. W1/W2 are still evaluated on every
forward pass, so their current learned values affect every batch.

The cache avoids repeatedly running the approximately RoBERTa-base-sized
emotion encoder during every epoch. In the current uploaded evaluation
artifact it is read from:

```text
/kaggle/input/datasets/cheikhmohamedahid/eat-encoder/cache/
goemotions_baseline_raw_aligned_fp16_v3.pt
```

### 3.3 Comparable protocol

The comparable baseline and EAT stages share protocol identifier
`bart_eat_comparable_v1` and the following settings:

| Setting | Baseline | EAT |
|---|---:|---:|
| Dataset and deterministic split | same | same |
| Split seed | 42 | 42 |
| Maximum source length | 256 | 256 |
| Maximum target length | 512 | 512 |
| `add_prefix_space` | false | false |
| Per-device training batch | 4 | 4 |
| Per-device evaluation batch | 4 | 4 |
| Gradient accumulation | 4 | 4 |
| Expected CUDA devices | 2 | 2 |
| Effective global batch | 32 | 32 |
| Precision | float16 | float16 |
| Maximum epochs | 30 | 40 |
| Early-stopping patience | 3 | 3 |
| Learning rate | 3e-5 | 3e-4 |
| Trainable parameters | all baseline BART | EAT W1/W2 only |

The differing epoch ceilings and learning rates are intentional because the
two stages optimize different parameter sets. The EAT run starts from the
completed baseline rather than from `facebook/bart-base`.

### 3.4 Reproducibility safeguards

The following safeguards were added:

- a run manifest containing the fully resolved configuration, dataset SHA-256,
  protocol signature, and software versions;
- an EAT preflight that refuses to train when the baseline manifest, dataset,
  split-affecting settings, batch settings, tokenizer settings, or relevant
  package versions differ;
- an exact baseline artifact path and an exact uploaded EAT checkpoint path;
- pinned `transformers==5.0.0`, `accelerate==1.13.0`, and
  `sacrebleu==2.6.0`;
- deterministic, matched test generation settings: 512 new tokens maximum,
  four beams, no sampling, repetition penalty 1.15, no-repeat trigram,
  length penalty 1.15, and early stopping;
- preservation of generated token IDs for exact output comparison.

### 3.5 Correctness fixes discovered during smoke testing

Several early runs were invalidated or made difficult to interpret by concrete
pipeline problems. The final code includes these corrections:

- `accepts_loss_kwargs=False` prevents Transformers 5 from applying an
  incompatible `num_items_in_batch` loss contract during gradient
  accumulation. Before this fix, logged gradients were zero and W1/W2 did not
  update even though minibatch losses varied.
- Raw BART text tokenization was restored. Pretokenization and inconsistent
  prefix-space behavior had changed the baseline token stream.
- Target length was standardized at 512. Earlier baseline/EAT evaluations had
  used different target or generation lengths.
- Transformers 5 boolean attention masks are interpreted correctly.
- Alpha zero takes the native configured BART attention backend, rather than a
  numerically different eager reimplementation.
- EAT checkpoint loading validates missing and unexpected state-dict keys while
  allowing only known tied-weight aliases.
- Early stopping loads the best validation-loss checkpoint, and checkpoint
  retention is limited to two.

After the gradient fix, the five-step diagnostic changed as follows:

- validation loss: 2.5762765 before to 2.5754101 after;
- relative W update: 0.0663313;
- observed W gradient norms: approximately 0.0051 to 0.0060.

This replaced the earlier diagnostic in which the before/after loss was
identical, relative W update was zero, and every reported gradient norm was
zero.

## 4. Evaluation and reporting changes

- Automatic evaluation now covers all 1,987 test examples.
- BERTScore uses `roberta-large`; SacreBLEU uses its conventional 0-100 scale;
  ROUGE-L F1 and Distinct-2 are also recorded.
- The best checkpoint validation loss is extracted from Trainer state rather
  than assumed from a printed training log.
- LLM judging is capped at 100 examples to control API usage.
- GPT-OSS and Qwen configurations were added after the older Groq Llama model
  became unsuitable for the workflow.
- Judge aggregation now requires sufficient completion and can reject an
  incomplete judge instead of silently averaging zeros or a handful of
  successful examples.
- A 72-row EAT signal report records all 6 layers x 12 heads, including
  \(r_h=\operatorname{mean}|\alpha S_h|/\operatorname{mean}|A_h|\).
- Dedicated alpha-zero evaluation and scoring configurations were added. The
  alpha-zero score intentionally omits the stored alpha=0.1 training loss.

## 5. Experimental results

### 5.1 Historical exploratory results (not paper-comparable)

The first five-epoch baseline was reported as:

| Metric | Historical baseline |
|---|---:|
| BERTScore F1 | 0.8630 |
| BLEU-4 | 0.0381 |
| ROUGE-L F1 | 0.2389 |
| Distinct-2 | 0.0500 |
| Reported validation loss | 4.1290 |

Its saved Trainer history later showed validation losses of 4.6816, 4.3366,
4.1094, 3.9738, and 3.9261 across epochs 1-5. This disagreement was one reason
to rebuild the comparison protocol.

An early contextual EAT run produced BERTScore 0.8537, BLEU-4 0.0193,
ROUGE-L F1 0.1617, Distinct-2 0.0619, and validation loss 8.6619. Its evaluation
loss appeared constant across epochs, and later diagnostics confirmed that
W1/W2 were not updating in the affected setup.

After gradient and loading corrections, but before the final comparable
protocol, another contextual alpha=0.1 run produced:

| Metric | Later exploratory EAT |
|---|---:|
| BERTScore F1 | 0.879774 |
| BLEU-4 under the older scoring implementation | 0.054791 |
| ROUGE-L F1 | 0.250569 |
| Distinct-2 | 0.063478 |
| Validation loss | 2.480316 |

This initially appeared better than the historical baseline, but it cannot be
used as causal evidence because target/generation settings, loss handling, and
metric implementations were not yet matched. Its Llama judge failed 100/100
requests, while GPT-OSS completed 99/100.

These historical BLEU values must not be compared numerically with the final
SacreBLEU values below because the implementation and scale changed.

### 5.2 Final controlled baseline versus EAT

Both systems were evaluated on the same ordered set of 1,987 questions and
references with identical deterministic generation settings.

| Metric | Comparable baseline | EAT, alpha=0.10 | EAT - baseline |
|---|---:|---:|---:|
| Validation loss (lower is better) | 0.777848 | **0.776799** | **-0.001048** |
| BERTScore F1 | **0.897188** | 0.896746 | -0.000441 |
| SacreBLEU, 0-100 | **19.088464** | 19.007005 | -0.081459 |
| ROUGE-L F1 | **0.361384** | 0.358171 | -0.003213 |
| Distinct-2 | **0.156938** | 0.156662 | -0.000276 |
| Average generated tokens | 90.7011 | 91.6492 | +0.9482 |
| Empty prediction rate | 0 | 0 | 0 |

The EAT validation loss improved by approximately 0.135%, but every reported
test generation-quality/diversity metric decreased slightly. This run is best
described as approximately tied with, or slightly worse than, the baseline; it
does not establish an EAT performance improvement.

### 5.3 GPT-OSS judging

The comparable baseline had 100/100 successful GPT-OSS judgments. EAT had
99/100. Restricting the comparison to the 99 common successful examples gives:

| Criterion | Baseline | EAT | Difference | EAT wins / ties / losses |
|---|---:|---:|---:|---:|
| Empathy | 3.1818 | 3.1111 | -0.0707 | 10 / 73 / 16 |
| Coherence | 3.6768 | 3.6465 | -0.0303 | 10 / 76 / 13 |
| Safety | 4.6364 | 4.5556 | -0.0808 | 7 / 79 / 13 |

The judge therefore also indicates a near tie with a small baseline advantage.

Qwen results are excluded: only 8/100 baseline and 5/100 EAT responses were
successfully parsed. Most failures contained long `<think>` output or malformed
and truncated JSON. Averages computed from those tiny successful subsets are
not reliable, and the strict aggregate correctly could not produce a valid
two-judge result.

### 5.4 EAT signal diagnostic

For the final comparable EAT checkpoint, over 50 batches and 400 examples:

- fixed alpha: 0.10;
- overall \(r_h\): 0.303533;
- overall mean absolute native attention score: 1.875780;
- overall mean absolute scaled emotion score: 0.569360;
- nonzero emotion-feature coverage: 0.956611;
- minimum head \(r_h\): 0.024168 at layer 3, head 6;
- maximum head \(r_h\): 1.819929 at layer 0, head 1.

Mean \(r_h\) declined by encoder depth:

| Encoder layer | Mean head r_h |
|---:|---:|
| 0 | 0.689656 |
| 1 | 0.559373 |
| 2 | 0.421892 |
| 3 | 0.167115 |
| 4 | 0.139922 |
| 5 | 0.110975 |

This confirms a material, nonzero emotion-branch signal, especially in the
lower encoder layers. It does not by itself establish that the signal improves
generation quality.

### 5.5 Alpha-zero causal sanity check

The trained EAT checkpoint was evaluated without retraining, with alpha set to
zero. It reproduced the comparable baseline automatic metrics and, more
decisively, its generated token sequences exactly:

```text
Examples compared: 1987
Different token sequences: 0
Exact recovery: True
```

At alpha=0.10, 1,061 of 1,987 decoded generations differed from the baseline;
926 were identical. The exact alpha-zero recovery demonstrates that:

- the correct fine-tuned baseline checkpoint was loaded;
- frozen BART weights remained unchanged during EAT training;
- tokenizer, split, and deterministic generation settings were aligned;
- the model reduces exactly to native BART when EAT is disabled;
- output differences at alpha=0.10 are attributable to the EAT branch.

This is strong evidence of causal influence and implementation correctness,
but not evidence of beneficial influence, because the controlled alpha=0.10
metrics were slightly worse overall.

## 6. Current conclusions and recommended next experiment

1. The contextual EAT implementation is functioning, receives gradients, and
   measurably changes encoder behavior and generated outputs.
2. The final comparison is technically aligned, and the alpha-zero result is a
   particularly strong invariance check.
3. The earlier apparent improvement cannot be used as paper evidence because
   it came from mismatched configurations and scoring.
4. The controlled alpha=0.10 run does not outperform the baseline.
5. Alpha changes the convex mixture after two softmax operations, so W1/W2
   trained at alpha=0.10 are not guaranteed to be optimal at another alpha.

For a rigorous alpha study, train fresh W1/W2 matrices from the same frozen
baseline for alpha values 0.025, 0.05, and 0.075, retaining the completed 0.10
run. Keep all other settings fixed, select alpha using a predeclared validation
metric rather than the test split, and evaluate the selected configuration once
on the test set. The selected experiment should then be repeated with multiple
random seeds before making a paper-level performance claim.

## 7. Main implementation locations

- Contextual extraction/alignment/cache:
  `src/eat_bart/data/contextual_emotion.py`
- Contextual batch construction: `src/eat_bart/data/collator.py`
- W1/W2 and emotion score: `src/eat_bart/modeling/eat_attention.py`
- Probability-mixture attention and masking:
  `src/eat_bart/modeling/eat_bart_attention.py`
- Native BART feature routing and gradient-accumulation contract:
  `src/eat_bart/modeling/eat_bart_forward.py`
- Baseline checkpoint and EAT checkpoint loading:
  `src/eat_bart/modeling/eat_bart_model.py`
- Freezing, optimizer construction, training, caching, and early stopping:
  `src/eat_bart/training/train.py`
- Manifest/preflight validation: `src/eat_bart/training/comparability.py`
- EAT signal diagnostic: `src/eat_bart/training/eat_signal.py`
- Comparable EAT training config: `configs/kaggle_encoder_eat_comparable.yaml`
- Comparable generation config:
  `configs/kaggle_encoder_eat_comparable_evaluate.yaml`
- Alpha-zero ablation config:
  `configs/kaggle_encoder_eat_comparable_alpha0_evaluate.yaml`

At the time of this report, the encoder-EAT branch test suite contains 73
passing tests.
