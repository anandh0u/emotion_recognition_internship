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

The demo page loads the emotion and animated-content checkpoints, shows dataset examples when local feature caches exist, and supports audio/image uploads.

For cloud deployment, use the root entrypoint:

```powershell
streamlit run streamlit_app.py
```

The deployed app can run in upload-only mode with the small checkpoints in `models/`. The large feature caches in `features/` are intentionally not committed to GitHub; when they are absent, dataset-sample browsing is disabled, but audio/image uploads still work.

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

The app downloads Wav2Vec2 and ViT backbones from Hugging Face on first use, so the first uploaded prediction can be slow.

The UI has two tasks:

- Emotion Recognition: recommended mode uses the visual head because it currently performs best on the test split.
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
