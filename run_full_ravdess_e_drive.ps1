param(
    [string]$WorkRoot = "E:\emotion_recognition_internship",
    [string]$DataRoot = "E:\emotion_recognition_data",
    [int]$Epochs = 40,
    [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$Python = Join-Path $WorkRoot ".venv311\Scripts\python.exe"
$DownloadDir = Join-Path $DataRoot "downloads\ravdess"
$RavdessRoot = Join-Path $DataRoot "raw\ravdess_full"
$ProcessedDir = Join-Path $DataRoot "processed\ravdess_full"
$LabelsCsv = Join-Path $DataRoot "labels_ravdess_full.csv"
$FeaturesFile = Join-Path $DataRoot "features\ravdess_full_embeddings.pt"
$ModelsDir = Join-Path $DataRoot "models\ravdess_full"
$ResultsDir = Join-Path $DataRoot "results\ravdess_full"

New-Item -ItemType Directory -Force -Path $WorkRoot, $DataRoot, $DownloadDir, $RavdessRoot, $ProcessedDir | Out-Null

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$ScriptBlock,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )
    Write-Output "=== $Name ==="
    & $ScriptBlock
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

if (!(Test-Path $Python)) {
    py -3.11 -m venv (Join-Path $WorkRoot ".venv311")
    Invoke-Step { & $Python -m pip install --upgrade pip setuptools wheel } "Install packaging tools"
    Invoke-Step { & $Python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu } "Install PyTorch"
    Invoke-Step { & $Python -m pip install -r (Join-Path $WorkRoot "requirements.txt") } "Install project requirements"
}

Push-Location $WorkRoot
try {
    if (!$SkipDownload) {
        Invoke-Step { & $Python src\download_ravdess_zenodo.py `
            --download-dir $DownloadDir `
            --extract-dir $RavdessRoot `
            --actors all `
            --retries 30 `
            --retry-wait 20 } "Download and extract full RAVDESS"
    }

    Invoke-Step { & $Python src\prepare_ravdess_manifest.py `
        --root $RavdessRoot `
        --processed-dir $ProcessedDir `
        --output $LabelsCsv `
        --split-strategy actor `
        --val-actors 17,18,19,20 `
        --test-actors 21,22,23,24 } "Prepare full RAVDESS manifest"

    Invoke-Step { & $Python src\precompute.py `
        --labels $LabelsCsv `
        --raw-dir $RavdessRoot `
        --output $FeaturesFile `
        --overwrite } "Precompute full RAVDESS embeddings"

    Invoke-Step { & $Python src\train.py `
        --features $FeaturesFile `
        --models-dir $ModelsDir `
        --results-dir $ResultsDir `
        --epochs $Epochs `
        --lr 1e-4 `
        --batch 16 } "Train full RAVDESS classifier"

    Invoke-Step { & $Python src\evaluate.py `
        --checkpoint (Join-Path $ModelsDir "best_model.pt") `
        --cache $FeaturesFile `
        --split all `
        --modality auto `
        --results-dir $ResultsDir `
        --batch 16 } "Evaluate full RAVDESS all splits"

    foreach ($Modality in @("audio", "visual", "fusion")) {
        Invoke-Step { & $Python src\evaluate.py `
            --checkpoint (Join-Path $ModelsDir "best_model.pt") `
            --cache $FeaturesFile `
            --split test `
            --modality $Modality `
            --results-dir (Join-Path $DataRoot "results\ravdess_full_$Modality") `
            --batch 16 } "Evaluate full RAVDESS $Modality"
    }
}
finally {
    Pop-Location
}
