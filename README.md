# Multimodal Emotion Recognition Internship Project

Late-fusion emotion recognition for audio + visual inputs using precomputed backbone embeddings.

## Project layout

```text
emotion_recognition_internship/
├── data/
│   ├── raw/          # SAVEE audio files and FER2013 data/images
│   └── labels.csv    # Unified multimodal manifest
├── features/         # Saved embeddings: features/all_embeddings.pt
├── models/           # Model checkpoints
├── results/          # Confusion matrix and plots
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── precompute.py
│   ├── train.py
│   ├── train_audio_wav2vec2.py
│   ├── evaluate.py
│   ├── zero_shot.py
│   ├── few_shot.py
│   └── demo_app.py
├── requirements.txt
└── README.md
```

## Data format

`data/labels.csv` should contain one row per multimodal example with these columns:

- `sample_id`: unique example id
- `split`: `train`, `val`, or `test`
- `label`: one of `anger`, `disgust`, `fear`, `happiness`, `neutral`, `sadness`, `surprise`
- `audio_path`: path to the audio file relative to `data/raw` or an absolute path
- `image_path`: path to an image file relative to `data/raw`, an absolute image path, or a FER2013 CSV row reference such as `fer2013.csv#row=123`

Example header:

```csv
sample_id,split,label,audio_path,image_path
DC_a01,train,anger,ALL\DC_a01.wav,fer2013.csv#row=0
```

## Installation

Conda is not required. On this machine, use the clean Python 3.11 virtual environment:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv311\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
.\.venv311\Scripts\python.exe -c "import torch; print(torch.__version__)"
```

For every command below, use `.\.venv311\Scripts\python.exe` instead of plain `python` if your terminal is not activated.

To activate it in PowerShell:

```powershell
.\.venv311\Scripts\Activate.ps1
```

If PowerShell blocks activation, direct commands like `.\.venv311\Scripts\python.exe src\train.py ...` still work.

## Dataset Audit

After adding or extracting data, run:

```powershell
.\.venv311\Scripts\python.exe src\audit_data.py
```

This writes `results/data_audit.json` and `results/data_audit.md`.

## Download Extra Kaggle Datasets

The project includes KaggleHub support for the animated multimodal dataset and SAVEE. KaggleHub uses your local Kaggle login/cache.

```powershell
python src/download_datasets.py --method kagglehub
```

To download only the animated dataset:

```powershell
python src/download_datasets.py --method kagglehub --dataset animated
```

Downloaded files are copied to `data/raw/kaggle/`. After download, build an extra manifest from the local files:

```powershell
python src/prepare_media_manifest.py --root data/raw/kaggle --output data/labels_extra.csv
```

Review `data/labels_extra.csv`, then merge useful rows into `data/labels.csv` before precomputing embeddings again.

## Animated Content Dataset

The animated dataset has a different target from the emotion model. It is a paired audio + `.npy` visual-array dataset with binary labels:

- `not_optimized`
- `optimized`

Keep it separate from `data/labels.csv` unless your guide asks you to do a single mixed-task experiment.

If `data/raw/animated.zip` exists, extract it to `data/raw/animated`, then build its manifest:

```powershell
mkdir data\raw\animated
tar -xf data\raw\animated.zip -C data\raw\animated
python src/prepare_animated_manifest.py --root data/raw/animated --output data/labels_animated.csv
```

Precompute and train the animated-content model separately:

```powershell
python src/precompute.py --labels data/labels_animated.csv --raw-dir data/raw --output features/animated_embeddings.pt
python src/train.py --features features/animated_embeddings.pt --models-dir models/animated --results-dir results/animated --epochs 30 --lr 1e-4 --batch 16
python src/evaluate.py --cache features/animated_embeddings.pt --checkpoint models/animated/best_model.pt --results-dir results/animated --split test
```

This does not overwrite the current emotion checkpoint unless you use the default `models/` and `features/all_embeddings.pt` paths.

## RAVDESS Audio-Video Emotion Dataset

RAVDESS is the preferred emotion dataset because its audio and video come from the same acted clip.

For the full RAVDESS run on this machine, use the `E:` drive. The `C:` drive does not have enough free space for the full dataset, extracted media, embeddings, and checkpoints.

Recommended full workflow:

```powershell
cd E:\emotion_recognition_internship
.\run_full_ravdess_e_drive.ps1
```

The full workflow downloads RAVDESS Video Speech Actor 01-24 directly from the official Zenodo record, so no Kaggle credentials are required for this path.

If you manually download and extract RAVDESS into `E:\emotion_recognition_data\raw\ravdess_full`, skip the download step:

```powershell
cd E:\emotion_recognition_internship
.\run_full_ravdess_e_drive.ps1 -SkipDownload
```

This script uses:

- Project copy: `E:\emotion_recognition_internship`
- Dataset/download storage: `E:\emotion_recognition_data`
- Full manifest: `E:\emotion_recognition_data\labels_ravdess_full.csv`
- Full embeddings: `E:\emotion_recognition_data\features\ravdess_full_embeddings.pt`
- Full checkpoints: `E:\emotion_recognition_data\models\ravdess_full`
- Full evaluation results: `E:\emotion_recognition_data\results\ravdess_full`

The full workflow uses an actor-independent split:

- Train actors: all actors except validation/test actors
- Validation actors: `17,18,19,20`
- Test actors: `21,22,23,24`

This is the result to report as the stronger real-world estimate, because test actors are unseen during training.

If `data/raw/ravdess.zip` is available, extract a safe subset first:

```powershell
mkdir data\raw\ravdess
tar -xf data\raw\ravdess.zip -C data\raw\ravdess Actor_09
```

Prepare audio/frame files and a 7-class manifest:

```powershell
python src/prepare_ravdess_manifest.py --root data/raw/ravdess --output data/labels_ravdess.csv
```

This writes extracted media under `data/processed/ravdess/` and creates `data/labels_ravdess.csv`. RAVDESS `calm` clips are mapped to `neutral` for this project's 7-class schema.

For actor-independent splitting with already extracted full RAVDESS files:

```powershell
python src/prepare_ravdess_manifest.py --root E:\emotion_recognition_data\raw\ravdess_full --processed-dir E:\emotion_recognition_data\processed\ravdess_full --output E:\emotion_recognition_data\labels_ravdess_full.csv --split-strategy actor --val-actors 17,18,19,20 --test-actors 21,22,23,24
```

Precompute and train:

```powershell
python src/precompute.py --labels data/labels_ravdess.csv --raw-dir data/raw --output features/ravdess_embeddings.pt
python src/train.py --features features/ravdess_embeddings.pt --models-dir models/ravdess --results-dir results/ravdess --epochs 40 --lr 1e-4 --batch 16
```

Current full RAVDESS actor-independent results:

- Deployed recommended audio SVM accuracy: `60.00%`
- Deployed recommended audio SVM weighted F1: `59.26%`
- Deployed recommended audio SVM UAR: `60.57%`
- Neural audio head accuracy on audio-bearing test samples: `50.42%`
- Neural fusion accuracy on audio+visual test samples: `43.33%`
- Visual-only test accuracy: `27.08%`
- Best audio SVM validation F1: `58.82%`

These numbers are lower than the earlier Actor_09-only experiment because this split tests on unseen actors. That makes it a better estimate of real-world generalization.

## Stronger Audio Fine-Tuning

The deployed SVM is the best current lightweight model, but the next accuracy jump should come from fine-tuning Wav2Vec2 directly on audio files. This machine currently has CPU-only PyTorch, so full fine-tuning is better suited to Colab, Kaggle GPU, or another CUDA machine.

Smoke test on this machine:

```powershell
.\.venv311\Scripts\python.exe src\train_audio_wav2vec2.py --labels E:\emotion_recognition_data\labels_ravdess_full.csv --output-dir E:\emotion_recognition_data\models\wav2vec2_smoke --epochs 1 --batch 1 --max-duration 1.0 --freeze-base --unfreeze-last-n 0 --limit-train 2 --limit-val 2 --limit-test 2 --no-save-model
```

Recommended GPU run:

```powershell
python src/train_audio_wav2vec2.py --labels E:\emotion_recognition_data\labels_ravdess_full.csv --output-dir E:\emotion_recognition_data\models\wav2vec2_emotion_full --epochs 8 --batch 4 --lr 2e-5 --max-duration 4.0 --freeze-feature-encoder --freeze-base --unfreeze-last-n 2
```

For a stronger final model, merge RAVDESS with CREMA-D, TESS, SAVEE, and EmoDB into one manifest using the same `sample_id,split,label,audio_path` columns, then train this script on the combined manifest with speaker-independent splits.

## 1) Precompute embeddings once

This extracts Wav2Vec2 and ViT features and stores them in `features/all_embeddings.pt`.

```powershell
python src/precompute.py --labels data/labels.csv --raw-dir data/raw --output features/all_embeddings.pt
```

## 2) Train the late-fusion classifier

```powershell
python src/train.py --epochs 30 --lr 1e-4 --batch 16 --wandb
```

## 3) Evaluate the checkpoint

```powershell
python src/evaluate.py --checkpoint models/best_model.pt --cache features/all_embeddings.pt --split test
```

This writes `results/confusion_matrix.png` and prints accuracy, weighted F1, and UAR.

To evaluate every split:

```powershell
python src/evaluate.py --checkpoint models/best_model.pt --cache features/all_embeddings.pt --split all
```

## 4) Run the browser demo

```powershell
streamlit run src/demo_app.py
```

The demo page loads the emotion and animated-content checkpoints, shows bundled RAVDESS sample files, and supports audio, image, and video uploads.

For cloud deployment, use the root entrypoint:

```powershell
streamlit run streamlit_app.py
```

The deployed app can run with the small checkpoints in `models/` plus the lightweight examples in `samples/`. The large feature caches in `features/` are intentionally not committed to GitHub; when they are absent, the app falls back to bundled sample media and extracts embeddings on demand.

## Deployment

Recommended target for this project is Streamlit Community Cloud because the demo is a Streamlit app.

Use these settings:

- Repository: `anandh0u/emotion_recognition_internship`
- Branch: `main`
- Main file path: `streamlit_app.py`
- Python: `3.11`

Deployment files included:

- `streamlit_app.py`
- `.streamlit/config.toml`
- `packages.txt`
- `runtime.txt`
- `models/best_model.pt`
- `models/animated/best_model.pt`
- `models/ravdess/best_model.pt`
- `models/ravdess/audio_svc.joblib`
- `samples/sample_manifest.csv`
- `samples/ravdess/`

The app downloads Wav2Vec2 and ViT backbones from Hugging Face on first use, so the first sample or uploaded prediction can be slow.

The UI has two tasks:

- Emotion Recognition: recommended mode uses the RAVDESS audio SVM because it currently performs best on the actor-independent test split; image and fusion modes still use the neural checkpoint.
- Animated Content Analysis: recommended mode uses the audio head because it currently performs best on the test split.

## 5) Zero-shot baseline

```powershell
python src/zero_shot.py --cache features/all_embeddings.pt
```

## 6) Few-shot prototypical baseline

```powershell
python src/few_shot.py --cache features/all_embeddings.pt --k 5
```

Use `--k 10` or `--k 20` for other support sizes.

## Notes

- `precompute.py` is meant to run once per dataset version. Training and evaluation only read `features/all_embeddings.pt`.
- The included `labels.csv` pairs SAVEE audio examples with same-label FER2013 rows because SAVEE and FER2013 are separate datasets rather than naturally paired recordings.
- `train.py` uses AdamW, cosine annealing, gradient clipping, and optional Weights & Biases logging.
- If your CSV uses different paths, keep them consistent and the scripts will resolve them relative to `data/raw`.
