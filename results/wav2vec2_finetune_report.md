# Wav2Vec2 Fine-Tuning Report

Project: Multimodal Emotion Recognition - Audio Agent  
Dataset: Full RAVDESS actor-independent audio split  
Evaluation date: 2026-06-07  
Model: `facebook/wav2vec2-base` fine-tuned with sequence-classification head  
Output directory: `E:\emotion_recognition_data\models\wav2vec2_emotion_full`

## Training Setup

- Train rows: 960 usable audio samples
- Validation rows: 240 usable audio samples
- Test rows: 240 usable audio samples
- Classes: anger, disgust, fear, happiness, neutral, sadness, surprise
- Epochs: 8
- Batch size: 4
- Learning rate: 2e-5
- Weight decay: 1e-2
- Gradient clipping: 1.0
- Max duration: 4.0 seconds
- Device: CPU
- Feature encoder frozen: yes
- Base model frozen: yes
- Last transformer layers unfrozen: 2

Command:

```powershell
.\.venv311\Scripts\python.exe src\train_audio_wav2vec2.py --labels E:\emotion_recognition_data\labels_ravdess_full.csv --output-dir E:\emotion_recognition_data\models\wav2vec2_emotion_full --epochs 8 --batch 4 --lr 2e-5 --max-duration 4.0 --freeze-feature-encoder --freeze-base --unfreeze-last-n 2
```

## Result

| Model | Accuracy | Weighted F1 | Weighted Precision | Weighted Recall | UAR | Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fine-tuned Wav2Vec2 audio agent | 53.33% | 49.34% | 55.01% | 53.33% | 50.45% | 1.2434 |
| Current deployed audio SVM | 60.00% | 59.26% | 60.70% | 60.00% | 60.57% | n/a |

Best validation weighted F1 during Wav2Vec2 fine-tuning: 56.03% at epoch 7.

## Decision

The fine-tuned Wav2Vec2 experiment is useful, but it should not replace the deployed audio SVM yet. The SVM trained on frozen Wav2Vec2 embeddings still gives the best actor-independent test performance and should remain the recommended audio agent in the Streamlit demo.

## Interpretation

The model improved during training, but validation performance did not transfer strongly enough to unseen test actors. Because this run was CPU-limited, only the final two transformer layers were unfrozen. That is a conservative setup and may underfit speaker-independent emotion features while still overfitting actor-specific patterns.

## Next Experiment

For a stronger paper-aligned audio agent:

1. Run on GPU.
2. Unfreeze the last 4 to 6 transformer layers.
3. Use audio augmentation such as noise, gain shift, and time masking.
4. Train on combined RAVDESS + CREMA-D + TESS + SAVEE audio manifests with speaker-independent splits.
5. Compare against the SVM using the same actor/speaker-independent test protocol.

