# Local Training Commands

Use these commands from the project folder:

```powershell
cd E:\emotion_recognition_internship
```

## 0) Check Environment

```powershell
.\.venv311\Scripts\python.exe -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
```

For Wav2Vec2 fine-tuning on this laptop, use CPU first. The RTX 3050 Laptop GPU has limited VRAM, so CUDA can crash during Wav2Vec2 with exit code `-1` or CUDA out-of-memory errors.

If you still want CUDA for short smoke tests or the lightweight feature-head models, install CUDA-enabled PyTorch:

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
.\run_local_audio_training.ps1 -Mode smoke -Device cpu
```

This checks that audio loading, labels, model loading, training, and evaluation all work.

## 3) RAVDESS Fine-Tuning Run

This is the conservative CPU local run:

```powershell
.\run_local_audio_training.ps1 -Mode ravdess -Epochs 8 -Batch 4 -LearningRate 2e-5 -UnfreezeLastN 2 -Device cpu
```

Current completed CPU result:

- Test accuracy: `53.33%`
- Weighted F1: `49.34%`
- UAR: `50.45%`

This did not beat the deployed audio SVM, which is still `60.00%` accuracy.

## 4) Multi-Dataset Fine-Tuning Run

This is the next stronger local experiment:

```powershell
.\run_local_audio_training.ps1 -Mode multi -Epochs 10 -Batch 4 -LearningRate 2e-5 -UnfreezeLastN 2 -Device cpu
```

Only try this tiny GPU version if you want to test CUDA. If it crashes, go back to the CPU command above:

```powershell
.\run_local_audio_training.ps1 -Mode multi -Epochs 3 -Batch 1 -LearningRate 1e-5 -MaxDuration 2.0 -UnfreezeLastN 1 -GradientAccumulationSteps 8 -Device cuda -Amp
```

## 5) Build All Agent Feature Caches

This builds CPU-safe Wav2Vec2/ViT embedding caches for the separate agents:

```powershell
.\run_agent_feature_build.ps1 -Mode all -Device cpu
```

Build one agent at a time:

```powershell
.\run_agent_feature_build.ps1 -Mode audio -Device cpu
.\run_agent_feature_build.ps1 -Mode vision -Device cpu
.\run_agent_feature_build.ps1 -Mode multimodal -Device cpu
.\run_agent_feature_build.ps1 -Mode animation -Device cpu
```

If a cache already exists and you intentionally want to rebuild it:

```powershell
.\run_agent_feature_build.ps1 -Mode vision -Device cpu -Overwrite
```

## 6) Train The Lightweight Feature-Head Models

These are much lighter than direct Wav2Vec2 fine-tuning:

```powershell
.\run_agent_feature_training.ps1 -Mode vision -Epochs 30 -Batch 16 -Device cpu
.\run_agent_feature_training.ps1 -Mode multimodal -Epochs 30 -Batch 16 -Device cpu
.\run_agent_feature_training.ps1 -Mode animation -Epochs 30 -Batch 16 -Device cpu
```

You can use CUDA here if the feature cache already exists, because the backbone is not being trained:

```powershell
.\run_agent_feature_training.ps1 -Mode vision -Epochs 30 -Batch 16 -Device cuda
```

## 7) One-Command Audio Runner

The same workflows are available through:

```powershell
.\run_local_audio_training.ps1 -Mode smoke -Device cpu
.\run_local_audio_training.ps1 -Mode manifest
.\run_local_audio_training.ps1 -Mode ravdess -Epochs 8 -Batch 4 -Device cpu
.\run_local_audio_training.ps1 -Mode multi -Epochs 10 -Batch 4 -Device cpu
```

## 8) Where Results Go

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

## 9) What To Deploy

Only replace the deployed audio SVM if a new model beats:

- Accuracy: `60.00%`
- Weighted F1: `59.26%`
- UAR: `60.57%`

Until then, keep `models\ravdess\audio_svc.joblib` as the recommended emotion model.
