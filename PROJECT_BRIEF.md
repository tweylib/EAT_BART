Project: decoder-placement Emotion-Aware Transformer using BART.

Goal:
Measure EAT in BART decoder self-attention for mental-health response generation
under the same controlled protocol as the completed placement comparison.

Models:
- Generator: facebook/bart-base, initialized from the comparable fine-tuned
  baseline checkpoint.
- Emotion features: frozen SamLowe/roberta-base-go_emotions hidden states.

Architecture:
- Modify decoder self-attention only.
- Keep encoder self-attention and decoder cross-attention native BART.
- Do NOT modify cross-attention.
- Freeze all BART parameters; train decoder EAT W1/W2 only.

Decoder causality:
- Build emotion states from decoder input prefixes with a triangular RoBERTa
  attention mask.
- Cache teacher-forced training features aligned to shifted decoder inputs.
- Recompute features from the live prefix during generation, after beam
  expansion/reordering; never read the reference response at inference time.

EAT attention:
- Formula: probability_mix.
- W1_s and W2_s: [num_heads, 768, 64], tied-orthogonal initialization.
- Alpha target: 0.05, warmed from 0.005 over five epochs.

Constraints:
- Code must run on Kaggle GPU.
- Use fp16 and gradient accumulation.
- Do not modify cross-attention or silently change the attention formula.
