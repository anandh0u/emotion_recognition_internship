# RAVDESS Evaluation Report

Project: Multimodal Emotion Recognition - Audio + Visual Late Fusion  
Dataset: RAVDESS partial subset  
Evaluation date: 2026-06-03  
Checkpoint: `models/ravdess/best_model.pt`  
Feature cache: `features/ravdess_embeddings.pt`

## Dataset Summary

- Total labelled samples: 115
- Train split: 79 samples
- Validation split: 15 samples
- Test split: 21 samples
- Classes: anger, disgust, fear, happiness, neutral, sadness, surprise
- Note: calm samples are mapped to neutral.

## Main Evaluation: Auto/Fusion Mode

| Split | Accuracy | Weighted F1 | UAR | Loss |
|---|---:|---:|---:|---:|
| Train | 98.73% | 98.74% | 98.90% | 0.2046 |
| Validation | 80.00% | 78.44% | 83.33% | 0.4745 |
| Test | 85.71% | 85.58% | 85.71% | 0.5747 |

## Test Evaluation By Modality

| Modality | Accuracy | Weighted F1 | UAR | Loss |
|---|---:|---:|---:|---:|
| Audio only | 80.95% | 79.93% | 80.95% | 0.9829 |
| Visual only | 95.24% | 95.10% | 95.24% | 0.4978 |
| Fusion | 85.71% | 85.58% | 85.71% | 0.5747 |

## Generated Files

- `results/ravdess/evaluation_all_metrics.json`
- `results/ravdess/confusion_matrix.png`
- `results/ravdess/confusion_matrix_train.png`
- `results/ravdess/confusion_matrix_val.png`
- `results/ravdess_audio/confusion_matrix.png`
- `results/ravdess_visual/confusion_matrix.png`
- `results/ravdess_fusion/confusion_matrix.png`

## Commands Used

```powershell
.\.venv311\Scripts\python.exe src\evaluate.py --checkpoint models\ravdess\best_model.pt --cache features\ravdess_embeddings.pt --split all --modality auto --results-dir results\ravdess --batch 16
.\.venv311\Scripts\python.exe src\evaluate.py --checkpoint models\ravdess\best_model.pt --cache features\ravdess_embeddings.pt --split test --modality audio --results-dir results\ravdess_audio --batch 16
.\.venv311\Scripts\python.exe src\evaluate.py --checkpoint models\ravdess\best_model.pt --cache features\ravdess_embeddings.pt --split test --modality visual --results-dir results\ravdess_visual --batch 16
.\.venv311\Scripts\python.exe src\evaluate.py --checkpoint models\ravdess\best_model.pt --cache features\ravdess_embeddings.pt --split test --modality fusion --results-dir results\ravdess_fusion --batch 16
```

## Interpretation

The RAVDESS partial subset performs best in visual-only mode, reaching 95.24% test accuracy. Audio-only performance is also strong at 80.95%. The fusion model reaches 85.71% test accuracy.

This result is useful for proving that the pipeline works, but it should be reported carefully because the current RAVDESS data is only a partial subset. For a stronger final result, the next step is to include more RAVDESS actors and use an actor-independent split.
