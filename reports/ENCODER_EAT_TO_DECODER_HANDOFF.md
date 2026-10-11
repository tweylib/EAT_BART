# Encoder EAT-BART Ablation: Results and Decoder Handoff

**Prepared:** 2026-10-11  
**Repository branch:** `ablation/encoder-eat`  
**Remote HEAD reviewed:** `fa9a318` (`Add causal emotion ablation evaluation`)  
**Purpose:** give a new Codex conversation enough technical and experimental
context to create a scientifically valid decoder-side ablation without
silently breaking comparability or introducing target leakage.

## 1. Executive summary

The completed work built and tested an encoder-self-attention EAT variant on
top of the same fine-tuned BART baseline used throughout the comparable
experiments. The final encoder variant used frozen BART, frozen
`SamLowe/roberta-base-go_emotions`, contextual RoBERTa hidden states aligned to
BART source tokens, layer/head-specific W1/W2 projections, and a probability
mixture of native and emotion attention.

The implementation was shown to be active and internally correct:

- W1/W2 received nonzero gradients and changed substantially.
- emotion-attention entropy decreased during training;
- the EAT branch changed generated sequences;
- setting alpha to zero recovered the native baseline token sequences exactly;
- aligned, shuffled, and zero-feature causal evaluations were implemented;
- a blinded pairwise judge was implemented with randomized A/B order and
  automatic ties for exactly identical outputs.

However, the final causal result was negative for the frozen encoder ablation.
Correctly aligned emotion features did not outperform shuffled emotion
features, and aligned EAT did not significantly outperform the baseline.
Therefore this experiment demonstrates causal influence of the EAT branch,
but not a beneficial causal effect of token-aligned emotion information.

The proposed decoder experiment is reasonable as a placement ablation, but it
must be treated as a new engineering problem. The existing contextual feature
pipeline is encoder/source-only. Merely changing
`modify_encoder_self_attention` to false and
`modify_decoder_self_attention` to true would not yield a valid decoder
experiment.

## 2. Stable reference artifacts and protocol

### 2.1 Dataset

```text
/kaggle/input/datasets/cheikhmohamedahid/eat-data/
finalMentalHealthDataset-question-response.csv
```

Columns are `question` and `response`. The deterministic split uses seed 42,
with validation and test fractions of 0.1 each.

The current 1,987-example test split has already influenced multiple
architecture and hyperparameter decisions. It should now be considered a
development/diagnostic set, not an untouched final test set. A paper-level
claim requires a new untouched holdout after the decoder/encoder choice is
locked, ideally with multiple random seeds.

### 2.2 Fine-tuned BART initialization

All comparable EAT variants start from:

```text
/kaggle/input/datasets/cheikhtidjanitweylib/
baseline-30-eps-new-model/models/bart_baseline_comparable
```

Do not fall back to `facebook/bart-base`. Do not initialize from an encoder-EAT
checkpoint when the purpose is to compare encoder and decoder placement. Both
placements must start independently from this same fine-tuned baseline.

### 2.3 Comparable preprocessing and generation

Keep these settings identical:

| Setting | Value |
|---|---:|
| BART tokenizer `add_prefix_space` | false |
| Maximum source length | 256 |
| Maximum target length | 512 |
| Train/eval per-device batch size | 4 / 4 |
| Gradient accumulation | 4 |
| Expected GPUs | 2 |
| Precision | float16 |
| Generation maximum new tokens | 512 |
| Beams | 4 |
| Sampling | false |
| Repetition penalty | 1.15 |
| No-repeat n-gram size | 3 |
| Length penalty | 1.15 |
| Generation early stopping | true |

The repository pins the principal comparison environment to Transformers
5.0.0, Accelerate 1.13.0, and SacreBLEU 2.6.0. Preserve the baseline manifest
and comparability checks.

## 3. Final encoder architecture

### 3.1 Emotion representations

The emotion classifier is:

```text
SamLowe/roberta-base-go_emotions
```

The 28 classifier logits are not the token features. The implementation uses
the final hidden layer of the underlying frozen RoBERTa encoder, so every
emotion token representation has dimension 768. RoBERTa is always in eval
mode, all its parameters are frozen, and feature extraction runs without
gradients.

The warning below was observed while loading the classifier and was confirmed
to be harmless:

```text
roberta.embeddings.position_ids | UNEXPECTED
```

### 3.2 BART/RoBERTa alignment

The code never relies blindly on equal token indices. It obtains character
offsets from both fast tokenizers. If the complete span tensors match, it uses
an identity fast path. Otherwise, it mean-pools all RoBERTa token states whose
character spans overlap a BART token span.

The invariant is:

```text
one contextual emotion representation per BART token
```

Special and padding positions are zero. A valid BART token that cannot be
aligned raises an error rather than receiving a guessed representation. This
ensures that emotion score S[i,j] refers to the same BART positions as native
attention score A[i,j].

Relevant code:

- `src/eat_bart/data/contextual_emotion.py`
- `src/eat_bart/data/collator.py`

### 3.3 Emotion score

An earlier intermediate W_E projection was removed. The retained formula is,
for each selected layer and attention head:

```text
Z1_h = E W1_h
Z2_h = E W2_h
S_h  = Z1_h Z2_h^T / sqrt(d_s)
```

W1 and W2 are layer-specific and head-specific. They are not shared between
layers or heads. In the final encoder run:

```text
E dimension = 768
d_s         = 64
W1/W2 shape per layer = [12, 768, 64]
```

With tied-orthogonal initialization, each head receives one orthogonally
initialized matrix W, then W1 and W2 are both copied from it. They remain two
independent parameters and can diverge after the first optimizer step.

Relevant code: `src/eat_bart/modeling/eat_attention.py`.

### 3.4 Probability mixture

The final formulation was:

```text
P_A = softmax(A + mask)
P_S = softmax(S + mask)
P   = (1 - alpha) P_A + alpha P_S
O   = P V
```

There is no extra softmax after mixing. Attention dropout is applied to the
final probability matrix before multiplying by V. The same key-validity mask
is applied to both branches. Transformers 5 boolean masks use `True` for an
allowed key, so the mask is inverted before `masked_fill`.

The final encoder experiment warmed alpha linearly from 0.005 to 0.05 over the
first five epochs. Alpha is a fixed buffer for probability-based formulas, not
an optimized parameter.

Relevant code: `src/eat_bart/modeling/eat_bart_attention.py`.

### 3.5 Freezing and cache

The completed encoder ablation froze all BART and RoBERTa weights and trained
only W1/W2. The aligned RoBERTa states were cached before W1/W2, so W1/W2 were
still recomputed and trained on every forward pass. A float16 full source
cache was estimated at about 1.29 GiB.

The causal evaluation artifact used:

```text
/kaggle/input/datasets/cheikhmohamedahid/a005-test/cache/
goemotions_baseline_raw_aligned_fp16_v3.pt
```

## 4. Important bugs and corrections already made

Do not reintroduce any of the following:

1. **Transformers 5 loss normalization:**
   `accepts_loss_kwargs=False` is required by the forwarding wrapper. Before
   this correction, logged gradient norms were zero and W1/W2 did not update.
2. **Raw text tokenization:** do not pretokenize BART input or change
   `add_prefix_space`; doing so changes the baseline token stream.
3. **Target length:** both baseline and EAT use 512, not the historical 128.
4. **Boolean mask semantics:** Transformers 5 boolean attention masks mark
   allowed positions with `True`.
5. **Alpha-zero native path:** alpha zero must call native BART attention so
   that it exactly reproduces the configured backend rather than a numerically
   different eager implementation.
6. **Checkpoint loading:** validate missing and unexpected keys, permitting
   only known tied-weight aliases.
7. **Kaggle paths:** `/kaggle/input` is read-only. New caches, models, and
   reports must be written under `/kaggle/working`.
8. **Evaluation artifacts:** automatic metric configs for shuffled/zero
   generation intentionally do not reuse a stored aligned validation loss.

After the loss-contract correction, a five-step smoke test produced nonzero
gradient norms around 0.005-0.006, a relative W update of about 0.0663, and a
small validation-loss change. A smoke test must again verify actual decoder W
updates before a long Kaggle run.

## 5. Encoder experiments performed

The following controlled ideas were tested, always from the comparable
fine-tuned BART baseline:

1. probability mixture with alpha 0.10 and LR 1e-4;
2. probability mixture with alpha 0.05 and LR 1e-4;
3. probability composition
   `P=(1-alpha)P_A + alpha(P_A @ P_S)`;
4. tied-orthogonal W1=W2 initialization, followed by independent training;
5. alpha warmup 0.005 to 0.05 and d_s increased from 32 to 64;
6. evaluation-only aligned, shuffled, zero-feature, and alpha-zero controls;
7. blinded pairwise judging of aligned versus shuffled and aligned versus the
   baseline.

Most d_s=32 runs stopped after epoch 4 because epoch 1 had the lowest
validation loss and patience was three. The final warmup/d_s=64 run disabled
early stopping and was allowed to run the configured 40 epochs.

## 6. Main empirical results

### 6.1 Comparable baseline

| Metric | Baseline |
|---|---:|
| BERTScore F1 | 0.8972 |
| SacreBLEU-4 | 19.0885 |
| ROUGE-L F1 | 0.3614 |
| Distinct-2 | 0.1569 |
| Validation loss | 0.7778 |
| GPT-OSS empathy | 3.1700 |
| GPT-OSS coherence | 3.6600 |
| GPT-OSS safety | 4.6300 |

### 6.2 Final encoder EAT: warmup, d_s=64, alpha=0.05

| Metric | Final encoder EAT | EAT - baseline |
|---|---:|---:|
| BERTScore F1 | 0.8971 | -0.0001 |
| SacreBLEU-4 | 19.0523 | -0.0362 |
| ROUGE-L F1 | 0.3607 | -0.0007 |
| Distinct-2 | 0.1566 | -0.0003 |
| Validation loss | 0.7754 | -0.0024 |
| GPT-OSS empathy | 3.1800 | +0.0100 |
| GPT-OSS coherence | 3.6900 | +0.0300 |
| GPT-OSS safety | 4.6700 | +0.0400 |

The automatic metrics were effectively tied and slightly favored the baseline
except for teacher-forced validation loss. The pointwise judge differences
were small and were not established as statistically reliable.

### 6.3 Learning diagnostic

The final diagnostic showed that training was active rather than stuck:

- normalized emotion-attention entropy decreased approximately
  `0.9825 -> 0.7513`, meaning P_S became more concentrated and less uniform;
- W parameters moved by roughly 42% in relative norm;
- on a 40-example diagnostic subset, shuffled minus aligned loss was only
  about +0.000396;
- earlier diagnostics likewise showed large W changes but negligible loss
  changes.

This proves optimization and a changing emotion distribution. It does not
prove that the learned distribution contains useful generation information.

### 6.4 Causal feature controls

The same trained final checkpoint was evaluated with no retraining:

| Condition | BERTScore | BLEU-4 | ROUGE-L | Distinct-2 | Avg. generated tokens |
|---|---:|---:|---:|---:|---:|
| Aligned EAT | 0.8971 | 19.0523 | 0.3607 | 0.1566 | 90.2979 |
| Shuffled features | 0.8970 | 19.1830 | 0.3607 | 0.1694 | 90.6220 |
| Zero features | 0.8970 | 19.2896 | 0.3610 | 0.1573 | 91.0035 |
| Baseline | 0.8972 | 19.0885 | 0.3614 | 0.1569 | not recorded here |

`zero` is not the same as alpha zero. With zero features, S is zero and P_S
is uniform over allowed keys, while alpha remains 0.05. Alpha zero removes the
branch and takes native BART attention.

The aligned variant did not outperform shuffled or zero features on the
automatic metrics.

### 6.5 Blinded pairwise causal judge

The pairwise judge randomizes which system is called A/B. Exactly identical
responses are automatic ties and use no API request.

#### Shuffled EAT (left) versus aligned EAT (right), first 100 examples

- 73 responses were textually identical;
- 27 required API judging;
- 4 API comparisons failed, leaving 96 total successful examples;
- empathy: shuffled 9, aligned 8, ties 79, exact sign-test p=1.0000;
- coherence: shuffled 8, aligned 9, ties 79, p=1.0000;
- safety: shuffled 2, aligned 2, ties 92, p=1.0000.

There was no detectable benefit from correct token-to-emotion alignment.

#### Baseline (left) versus aligned EAT (right), first 100 examples

- 67 responses were textually identical;
- 33 required API judging;
- 4 API comparisons failed, leaving 96 total successful examples;
- empathy: baseline 8, EAT 11, ties 77, p=0.6476;
- coherence: baseline 8, EAT 9, ties 79, p=1.0000;
- safety: baseline 0, EAT 5, ties 91, p=0.0625.

The directions slightly favored EAT, especially safety, but none reached the
conventional 0.05 threshold. Safety used only five non-tied judgments and the
four missing comparisons could materially change it. These are exploratory
signals, not proof.

### 6.6 Defensible encoder conclusion

The encoder EAT branch clearly affected attention and sometimes changed
generation, but the frozen-BART encoder-only ablation did not demonstrate a
statistically significant improvement over the baseline or a benefit from
correctly aligned emotion features. It should be reported as a negative
ablation, not as proof of improved empathy.

## 7. Why decoder EAT is not a configuration-only change

### 7.1 The present contextual cache contains only source questions

For `goemotions_contextual`, `EATBartDataCollator` tokenizes target responses
normally and intentionally does not build `decoder_emotion_features`. The
cache is keyed by source-question text and is aligned to BART encoder tokens.

`evaluate.py` also loads question features only. During generation it
precomputes the EAT encoder and calls `model.generate`; it never creates or
updates decoder emotion features.

Consequently, setting:

```yaml
modify_encoder_self_attention: false
modify_decoder_self_attention: true
```

on the current baseline-loading path would patch decoder modules but would not
supply them with contextual emotion features during normal training and
generation.

### 7.2 Full-reference target features would leak the future

During teacher forcing, the complete reference response is available. It
would be easy to run bidirectional RoBERTa over that complete response and
align its states to decoder tokens. That is not a valid autoregressive
experiment: the feature for decoder position t would contain information from
future reference tokens, while those tokens do not exist at inference.

This could create an artificially strong training/evaluation loss and then
fail during free generation. Do not use full-target bidirectional RoBERTa
states as decoder features unless the experiment is explicitly labeled a
teacher-forced oracle, which is not comparable to generated-response quality.

### 7.3 Valid decoder features must be causal and available at inference

For decoder self-attention, the feature at position t may depend only on the
generated prefix available at t. Training and generation must use the same
definition.

With the current frozen bidirectional GoEmotions RoBERTa, the most literal
valid design is prefix conditioning:

1. build a RoBERTa representation from the response prefix available at each
   decoder position;
2. align that prefix representation to the corresponding BART decoder token;
3. shift features exactly as decoder inputs are shifted;
4. reproduce the same operation incrementally during generation.

Naively running RoBERTa once for every prefix is very expensive. A possible
research implementation is to apply a causal triangular attention mask to the
frozen RoBERTa in one training pass, but this must first be verified
empirically: RoBERTa is trained bidirectionally, its API must accept the mask,
and its incremental generation behavior must match the training features.

If a causal GoEmotions representation cannot be implemented faithfully within
the compute budget, decoder self-attention is not directly comparable under
the promised “same GoEmotions setup.” A source-emotion modification of decoder
cross-attention is a different, potentially valid architecture, but it must be
named and formulated as cross-attention EAT rather than decoder self-attention
EAT.

### 7.4 Incremental generation and KV-cache shape

During teacher forcing, decoder self-attention commonly has shape L by L.
During cached autoregressive generation, the query length is usually 1 while
the key/value length is the accumulated prefix length t.

The current `EmotionInteraction` computes a square matrix from the entire
feature tensor. A decoder implementation must produce or select emotion scores
of shape:

```text
[batch * beams, heads, query_length, key_length]
```

At an incremental step this will normally be `[B*beams, H, 1, t]`, not
`[B*beams, H, t, t]`. The final query row must correspond exactly to the
current decoder query. Tests must cover the first step, cached later steps,
variable prefix lengths, and padding.

### 7.5 Beam search

Beam search expands and reorders decoder sequences. Decoder emotion state must
be expanded and reordered with the same beam indices. A single
batch-level/static feature tensor is not sufficient. Test at least:

- greedy generation (`num_beams=1`);
- the comparable setting (`num_beams=4`);
- a case where beams reorder after the first step;
- equality with native BART at alpha zero.

### 7.6 Causal mask

P_S must receive the same decoder causal/key-validity mask as P_A. No decoder
query may place emotion probability on a future token. The Transformers 5
boolean-mask convention documented above still applies.

## 8. Code locations that require decoder-specific work

The next conversation should audit and modify at least these locations:

1. `src/eat_bart/modeling/eat_bart_model.py`
   - `load_eat_bart_from_baseline_checkpoint` is currently hard-coded to patch
     encoder self-attention only. Add explicit placement flags and validate
     exactly which modules were patched.
2. `src/eat_bart/training/train.py`
   - `freeze_bart_for_eat_only_training` is hard-coded to unfreeze encoder
     W1/W2 only;
   - the run-manifest stage is hard-coded as `encoder_eat`;
   - contextual cache construction currently processes source questions only.
3. `src/eat_bart/data/collator.py`
   - contextual target features are deliberately absent;
   - any decoder features must be causal, aligned, and shifted with
     `decoder_input_ids`.
4. `src/eat_bart/modeling/eat_bart_forward.py`
   - routing accepts `decoder_emotion_features`, but `generate()` does not
     currently build/update them per step or beam.
5. `src/eat_bart/modeling/eat_bart_attention.py`
   - verify incremental query/key dimensions and select the proper emotion
     score row under KV caching;
   - preserve the causal mask and native alpha-zero path.
6. `src/eat_bart/training/evaluate.py`
   - currently prepares source features and optionally precomputes only the
     encoder;
   - decoder generation needs a custom, tested feature-update path.
7. `src/eat_bart/training/eat_signal.py`
   - diagnostics are encoder-specific and must be generalized or replaced for
     decoder layers.
8. `src/eat_bart/training/eat_learning_diagnostic.py`
   - aligned/shuffled analysis and generation hooks assume encoder features.
9. `src/eat_bart/training/emotion_ablation.py`
   - the tensor transforms are reusable, but evaluation must apply them to
     valid decoder features without moving padding/special positions.
10. `src/eat_bart/modeling/patch_bart.py`
    - decoder self-attention patching exists, while cross-attention is
      intentionally unsupported. Do not silently patch cross-attention using
      the square self-attention implementation.

## 9. Recommended decoder experimental protocol

### 9.1 Preserve a clean placement comparison

The first decoder run should intentionally preserve the final encoder setup
where logically possible:

- initialize from the same `bart_baseline_comparable` checkpoint;
- frozen BART;
- frozen GoEmotions model;
- decoder self-attention modified, encoder self-attention native;
- W1/W2 only trainable;
- probability mixture, not probability composition;
- d_s=64;
- tied-orthogonal W1/W2 initialization;
- alpha warmup 0.005 to 0.05 over five epochs;
- W1/W2 LR 1e-4;
- same split, seed, batches, precision, tokenizer, lengths, and generation;
- 40-epoch ceiling if compute permits;
- do not use early stopping for the first strict comparison if the encoder
  reference used all 40 epochs, but save/evaluate every epoch and predeclare
  the checkpoint-selection rule.

The only intended conceptual differences should be attention placement and
the unavoidable causal construction of decoder emotion features. Document
that construction as part of the model definition, because it is a material
difference from the encoder's one-pass source features.

### 9.2 Required controls

For the trained decoder checkpoint, retain:

1. alpha-zero generation, which must reproduce native BART token IDs exactly;
2. aligned versus deterministically shuffled decoder features;
3. zero decoder features, recognizing that this creates a uniform P_S while
   retaining alpha;
4. exact generated-token comparison to the baseline;
5. blinded aligned-versus-shuffled and aligned-versus-baseline pairwise
   judging on the same ordered examples.

If decoder generation features cannot be shuffled consistently across steps
and beams, the causal claim is not ready.

### 9.3 Preflight before a full Kaggle run

Run a small deterministic smoke test and require all of the following:

- only decoder W1/W2 have `requires_grad=True`;
- all BART and RoBERTa tensors remain unchanged;
- every decoder layer contains EAT, and no encoder/cross-attention layer does;
- nonzero W gradients after backward;
- nonzero relative W update after several optimizer steps;
- loss or outputs change when alpha is nonzero;
- alpha zero exactly recovers baseline logits/token IDs within the established
  deterministic tolerance;
- teacher-forced and incremental first-step emotion-score shapes are correct;
- greedy and four-beam generation complete without feature/beam misalignment;
- no reference-response content is used during free generation;
- no decoder feature at position t depends on reference tokens after t.

Do not start a long run merely because minibatch loss fluctuates. Minibatch
composition caused changing losses even when the earlier broken setup had
zero gradients.

### 9.4 Decision rule

Automatic overlap metrics are secondary because the intended improvement is
empathetic response quality. Still report all metrics with the unchanged
pipeline.

The decoder placement should be considered promising only if:

1. aligned decoder EAT beats shuffled decoder EAT in blinded empathy judging;
2. aligned decoder EAT beats the same baseline checkpoint;
3. coherence and safety do not degrade;
4. results are not driven only by a handful of non-tied examples;
5. the result is repeated on an untouched holdout and preferably multiple
   seeds.

If aligned and shuffled remain tied, the experiment shows a generic attention
perturbation rather than useful use of emotion-token correspondence.

## 10. Existing evaluation infrastructure to reuse

The repository already includes:

- deterministic aligned/shuffled/zero feature transforms:
  `src/eat_bart/training/emotion_ablation.py`;
- blinded pairwise judging:
  `src/eat_bart/training/pairwise_judge.py` and
  `scripts/judge_pairwise.py`;
- automatic metrics: `scripts/score_generations.py`;
- pointwise judges: `scripts/judge_generations.py`;
- manifest and comparability checks:
  `src/eat_bart/training/comparability.py`;
- exact generated token IDs in generation CSV output.

The pairwise judge requires `GROQ_API_KEY` in the environment. In Kaggle:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["GROQ_API_KEY"] = UserSecretsClient().get_secret("GROQ_API_KEY")
```

Do not print the secret. A missing key previously caused every non-identical
pair to fail while identical pairs were still reported as automatic ties.

## 11. Current repository status caution

At the time this handoff was written, the remote branch HEAD was `fa9a318`.
The local working tree also contained uncommitted Qwen judge-format work:

- migration from deprecated `qwen/qwen3.6-27b` to
  `qwen/qwen3.8-27b`;
- hidden reasoning;
- strict JSON-schema output;
- one-example smoke and uploaded-artifact configs.

Those local changes had 114 passing mocked/local tests but had not been live
API-tested or committed/pushed. A new conversation should run `git status`
before editing and must not discard or accidentally mix those changes into the
decoder architecture commit. GPT-OSS was the usable judge for the reported
pairwise decoder-preparation decisions.

## 12. Suggested opening instruction for the new Codex conversation

```text
Read reports/ENCODER_EAT_TO_DECODER_HANDOFF.md completely before editing.
Create a decoder-self-attention EAT ablation from the same fine-tuned
bart_baseline_comparable checkpoint and preserve the final encoder experiment's
probability-mixture, GoEmotions, tied-orthogonal, d_s=64, alpha-warmup, data,
and evaluation settings. First design and test a causal decoder emotion-feature
pipeline that is available during autoregressive beam generation. Do not use
full-reference target RoBERTa states, do not patch cross-attention silently,
and do not start a full Kaggle run until alpha-zero exact recovery, KV-cache
shapes, beam reordering, freezing, and nonzero W updates all pass.
```

