# Full RAVDESS Evaluation Report

Project: Multimodal Emotion Recognition - Audio + Visual Late Fusion  
Dataset: Full RAVDESS Video Speech Actor 01-24  
Evaluation date: 2026-06-04  
Neural checkpoint: `models/ravdess/best_model.pt`  
Deployed audio classifier: `models/ravdess/audio_svc.joblib`  
Feature cache: `E:\emotion_recognition_data\features\ravdess_full_embeddings.pt`

## Dataset Summary

- Downloaded actor zip files: 24
- Raw video files: 2880
- Extracted visual frames: 2880
- Extracted audio files: 1440
- Classes: anger, disgust, fear, happiness, neutral, sadness, surprise
- Calm clips are mapped to neutral for the 7-class project schema.

## Split Strategy

The full experiment uses an actor-independent split:

- Train actors: all actors except validation/test actors
- Validation actors: 17, 18, 19, 20
- Test actors: 21, 22, 23, 24

This is stricter than a random split because the test actors are unseen during training.

## Main Evaluation: Auto/Fusion Mode

| Split | Accuracy | Weighted F1 | UAR | Loss |
|---|---:|---:|---:|---:|
| Train | 100.00% | 100.00% | 100.00% | 0.0113 |
| Validation | 51.46% | 50.19% | 49.93% | 1.8906 |
| Test | 35.42% | 34.38% | 35.49% | 2.7310 |

## Test Evaluation By Modality

| Modality | Accuracy | Weighted F1 | UAR | Notes |
|---|---:|---:|---:|---|
| Deployed audio SVM | 60.00% | 59.26% | 60.57% | Best current recommended mode |
| Fine-tuned Wav2Vec2 audio agent | 53.33% | 49.34% | 50.45% | Tested on 240 actor-independent audio samples; not deployed because it underperforms the SVM |
| Neural audio head | 50.42% | 49.84% | 51.93% | Evaluated only on audio-bearing test samples |
| Neural visual head | 27.08% | 23.74% | 27.53% | Weakest branch |
| Neural fusion head | 43.33% | 43.27% | 43.01% | Evaluated only on audio+visual test samples |
| Neural auto/fusion over all test records | 35.42% | 34.38% | 35.49% | Includes visual-only clips |

## Interpretation

The full RAVDESS result is a more realistic estimate than the earlier Actor_09-only experiment. The earlier result was high because training and testing were too close in actor distribution. The full actor-independent split shows that the neural late-fusion head overfits the training actors and generalizes weakly to unseen actors.

The best deployed recommendation is currently the audio SVM, trained on frozen Wav2Vec2 embeddings. It improves the recommended emotion mode from roughly 50% neural audio accuracy to 60% actor-independent test accuracy.

Direct Wav2Vec2 fine-tuning was also tested on the full actor-independent audio split. It reached 56.03% best validation weighted F1, but only 49.34% weighted F1 on the test split, so it remains an experimental audio agent rather than the deployed recommendation.

## Next Accuracy Improvements

- Use multiple frames per video instead of one middle frame.
- Fine-tune Wav2Vec2 and ViT lightly instead of using frozen embeddings only.
- Add stronger regularization, early stopping, and lower training epochs to reduce overfitting.
- Use class-balanced sampling and actor-balanced validation.
- Try audio-first fusion, since audio currently generalizes better than visual.
- Add augmentation: SpecAugment/noise for audio and crop/color/flip for images.

## Generated Files

- `E:\emotion_recognition_data\results\ravdess_full\evaluation_all_metrics.json`
- `E:\emotion_recognition_data\results\ravdess_full_audio\evaluation_metrics.json`
- `E:\emotion_recognition_data\results\ravdess_full_visual\evaluation_metrics.json`
- `E:\emotion_recognition_data\results\ravdess_full_fusion\evaluation_metrics.json`
- `E:\emotion_recognition_data\models\ravdess_full\best_model.pt`
- `models\ravdess\best_model.pt`
- `models\ravdess\audio_svc.joblib`
- `results\ravdess\audio_svc_metrics.json`
- `results\wav2vec2_finetune_report.md`
