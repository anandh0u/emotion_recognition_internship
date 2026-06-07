# Local Training Commands

Use these commands from the project folder:

```powershell
cd E:\emotion_recognition_internship
```

## 0) Check Environment

```powershell
.\.venv311\Scripts\python.exe -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
```

If CUDA prints `False` but `nvidia-smi` works, install CUDA-enabled PyTorch:

```powershell
.\.venv311\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
.\.venv311\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
.\.venv311\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## 1) Rebuild The Combined Audio Manifest

This uses the audio datasets already on this machine:

- Full RAVDESS audio manifest: `E:\emotion_recognition_data\labels_ravdess_full.csv`
- SAVEE: `data\raw\ALL`
- TESS: `E:\emotion_recognition_data\raw\TESS`
- CREMA-D: `E:\emotion_recognition_data\raw\CREMA-D\AudioMP3`

```powershell
.\.venv311\Scripts\python.exe src\prepare_audio_emotion_manifest.py --manifest E:\emotion_recognition_data\labels_ravdess_full.csv --root data\raw\ALL --root E:\emotion_recognition_data\raw\TESS --root E:\emotion_recognition_data\raw\CREMA-D\AudioMP3 --output E:\emotion_recognition_data\labels_audio_multi.csv --report results\multidataset_manifest_report.md
.\.venv311\Scripts\python.exe src\setup_agent_workspaces.py
```

Expected current size: `12162` audio rows.

## 2) Smoke Test The Fine-Tuning Script

Run this before any long training job:

```powershell
.\.venv311\Scripts\python.exe src\train_audio_wav2vec2.py --labels E:\emotion_recognition_data\labels_ravdess_full.csv --output-dir E:\emotion_recognition_data\models\wav2vec2_smoke --epochs 1 --batch 1 --max-duration 1.0 --freeze-base --unfreeze-last-n 0 --limit-train 2 --limit-val 2 --limit-test 2 --no-save-model
```

This checks that audio loading, labels, model loading, training, and evaluation all work.

## 3) RAVDESS Fine-Tuning Run

This is the conservative local run:

```powershell
.\.venv311\Scripts\python.exe src\train_audio_wav2vec2.py --labels E:\emotion_recognition_data\labels_ravdess_full.csv --output-dir E:\emotion_recognition_data\models\wav2vec2_emotion_full --epochs 8 --batch 4 --lr 2e-5 --max-duration 4.0 --freeze-feature-encoder --freeze-base --unfreeze-last-n 2
```

Current completed CPU result:

- Test accuracy: `53.33%`
- Weighted F1: `49.34%`
- UAR: `50.45%`

This did not beat the deployed audio SVM, which is still `60.00%` accuracy.

## 4) Multi-Dataset Fine-Tuning Run

This is the next stronger local experiment:

```powershell
.\.venv311\Scripts\python.exe src\train_audio_wav2vec2.py --labels E:\emotion_recognition_data\agents\audio\manifests\labels_audio_multi.csv --output-dir E:\emotion_recognition_data\agents\audio\models\wav2vec2_audio_multi --epochs 10 --batch 4 --lr 2e-5 --max-duration 4.0 --freeze-feature-encoder --freeze-base --unfreeze-last-n 2
```

If you have CUDA/GPU, increase the useful training capacity:

```powershell
.\.venv311\Scripts\python.exe src\train_audio_wav2vec2.py --labels E:\emotion_recognition_data\agents\audio\manifests\labels_audio_multi.csv --output-dir E:\emotion_recognition_data\agents\audio\models\wav2vec2_audio_multi_gpu --epochs 12 --batch 2 --lr 1e-5 --max-duration 4.0 --freeze-feature-encoder --freeze-base --unfreeze-last-n 4 --gradient-accumulation-steps 4 --amp --device cuda
```

## 5) One-Command Runner

The same workflows are available through:

```powershell
.\run_local_audio_training.ps1 -Mode smoke
.\run_local_audio_training.ps1 -Mode manifest
.\run_local_audio_training.ps1 -Mode ravdess -Epochs 8 -Batch 4
.\run_local_audio_training.ps1 -Mode multi -Epochs 10 -Batch 4
```

For GPU:

```powershell
.\run_local_audio_training.ps1 -Mode multi -Epochs 12 -Batch 2 -LearningRate 1e-5 -UnfreezeLastN 4 -GradientAccumulationSteps 4 -Device cuda -Amp
```

## 6) Where Results Go

Each run writes:

- `training_summary.json`
- `confusion_matrix.png`
- saved Hugging Face model folder under the chosen `--output-dir`

Example:

```text
E:\emotion_recognition_data\agents\audio\models\wav2vec2_audio_multi\training_summary.json
E:\emotion_recognition_data\agents\audio\models\wav2vec2_audio_multi\confusion_matrix.png
E:\emotion_recognition_data\agents\audio\models\wav2vec2_audio_multi\best_model\
```

## 7) What To Deploy

Only replace the deployed audio SVM if a new model beats:

- Accuracy: `60.00%`
- Weighted F1: `59.26%`
- UAR: `60.57%`

Until then, keep `models\ravdess\audio_svc.joblib` as the recommended emotion model.
