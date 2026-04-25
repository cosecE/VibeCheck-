# RAVDESS Emotion Classification: Frozen vs Fine-tuned Wav2Vec2

A controlled comparison of using Wav2Vec2 as a frozen feature extractor versus fine-tuning it for speech emotion classification on RAVDESS. The experimental variable is exactly one thing: whether Wav2Vec2's weights update during training. The classifier head architecture, dataset splits, evaluation metrics, and seeds are all held constant.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download RAVDESS

Get the audio-only speech subset from Zenodo: https://zenodo.org/record/1188976 (file `Audio_Speech_Actors_01-24.zip`, 1,440 .wav files). Unzip into `./ravdess/` (the structure inside doesn't matter — we glob recursively).

Expected layout:
```
ravdess/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   ├── 03-01-01-01-01-02-01.wav
│   └── ...
├── Actor_02/
└── ...
```

## Run

```bash
# 1) One-time: cache Wav2Vec2 embeddings for all clips (~10-15 min on GPU)
python extract_embeddings.py

# 2) Train the frozen baseline (~1 minute, runs on tiny cached features)
python train_frozen.py

# 3) Fine-tune Wav2Vec2 end-to-end (~30-60 min on GPU, much longer on CPU)
python train_finetune.py

# 4) Side-by-side comparison
python compare.py
```

After everything finishes, look in `./results/` for:
- `frozen_metrics.json`, `finetuned_metrics.json` — raw metrics
- `frozen_confusion.png`, `finetuned_confusion.png` — confusion matrices
- `frozen_curves.png`, `finetune_curves.png` — training curves
- `comparison.png` — per-class F1 bar chart comparing the two conditions

## Project structure

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters in one place |
| `data.py` | RAVDESS parsing, speaker-independent splits, dataset classes |
| `models.py` | Classifier head + full Wav2Vec2 wrapper |
| `evaluation.py` | Metrics, confusion matrix, training curves |
| `extract_embeddings.py` | One-shot script to cache frozen embeddings |
| `train_frozen.py` | Trains classifier head on cached embeddings |
| `train_finetune.py` | Full end-to-end fine-tuning |
| `compare.py` | Side-by-side comparison of the two conditions |

## Methodology notes

### Speaker-independent splits
Actors 1–18 → train, 19–21 → val, 22–24 → test. No actor appears in more than one split. This is essential for honest evaluation; speaker-dependent splits inflate accuracy by 10–30% on RAVDESS because models learn speaker-specific quirks rather than emotion-relevant features.

### Identical classifier head
Both conditions use the same MLP head architecture (768 → 256 → ReLU → Dropout → 8). The only difference is whether Wav2Vec2 updates.

### Fine-tuning details
- CNN feature encoder is frozen (`freeze_feature_encoder()`), standard practice.
- Differential learning rates: `5e-5` for the encoder, `1e-3` for the head.
- Linear warmup over the first 10% of training steps.
- Gradient clipping at norm 1.0.
- Early stopping on validation macro F1 (overfitting risk is high — 95M params on ~1000 training clips).

### Macro F1 as the headline metric
Class imbalance matters here: `neutral` only has half as many samples as the other emotions (no "strong intensity" version). Macro F1 gives equal weight to each class, while accuracy and weighted F1 over-reward correct predictions on the majority classes. Report all three but emphasize macro F1.

### Reproducibility
A single seed (`SEED = 42` in `config.py`) is set across PyTorch and CUDA. For a more rigorous study, run with multiple seeds (or different actor splits) and report mean ± std — this is straightforward to add by wrapping `main()` in a loop.

## Hardware

- **Frozen condition**: trivial. Will run on CPU in seconds once embeddings are cached.
- **Embedding extraction**: needs Wav2Vec2 forward passes on 1,440 clips. ~15 min on a modern GPU, ~1–2 hours on CPU.
- **Fine-tuning**: needs at least 8GB GPU VRAM (16GB+ recommended). On Colab/Kaggle T4 GPUs this works fine. On CPU, fine-tuning is impractical (many hours per epoch).

## Suggested extensions

If you have time after the main two-way comparison:

1. **LoRA condition** — add a third condition using parameter-efficient fine-tuning. Use Hugging Face's `peft` library to wrap the encoder. Updates ~0.5% of the params, often nearly matches full fine-tuning, and dramatically reduces overfitting risk.
2. **Multi-seed runs** — wrap `main()` in a loop over 3–5 seeds, report mean ± std. Makes the comparison statistically meaningful instead of anecdotal.
3. **Per-actor breakdown** — analyze test-set results by individual held-out actor. High variance across actors signals that the model is brittle to speaker characteristics.
