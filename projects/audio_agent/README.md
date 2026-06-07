# Audio Agent

Emotion recognition from speech/audio.

## Data

Manifest:

```text
E:\emotion_recognition_data\agents\audio\manifests\labels_audio_multi.csv
```

Current datasets:

- RAVDESS
- SAVEE
- TESS
- CREMA-D

## Train

After installing CUDA-enabled PyTorch, use the GPU command:

```powershell
cd E:\emotion_recognition_internship
.\.venv311\Scripts\python.exe src\train_audio_wav2vec2.py --labels E:\emotion_recognition_data\agents\audio\manifests\labels_audio_multi.csv --output-dir E:\emotion_recognition_data\agents\audio\models\wav2vec2_audio_multi_gpu --epochs 10 --batch 2 --lr 1e-5 --max-duration 4.0 --freeze-feature-encoder --freeze-base --unfreeze-last-n 4 --gradient-accumulation-steps 4 --amp --device cuda
```

For a smaller local test:

```powershell
.\.venv311\Scripts\python.exe src\train_audio_wav2vec2.py --labels E:\emotion_recognition_data\agents\audio\manifests\labels_audio_multi.csv --output-dir E:\emotion_recognition_data\agents\audio\models\wav2vec2_audio_smoke --epochs 1 --batch 1 --max-duration 1.0 --freeze-base --unfreeze-last-n 0 --limit-train 2 --limit-val 2 --limit-test 2 --no-save-model
```

## Output

```text
E:\emotion_recognition_data\agents\audio\models\
E:\emotion_recognition_data\agents\audio\results\
```
