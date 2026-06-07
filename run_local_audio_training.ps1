param(
    [ValidateSet("manifest", "smoke", "ravdess", "multi")]
    [string]$Mode = "smoke",

    [int]$Epochs = 1,
    [int]$Batch = 1,
    [double]$LearningRate = 2e-5,
    [double]$MaxDuration = 4.0,
    [int]$UnfreezeLastN = 2,
    [int]$GradientAccumulationSteps = 1,
    [ValidateSet("cpu", "cuda", "auto")]
    [string]$Device = "cpu",
    [switch]$Amp,
    [switch]$NoSaveModel
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv311\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$DataRoot = "E:\emotion_recognition_data"
$AgentRoot = Join-Path $DataRoot "agents"
$RavdessLabels = Join-Path $DataRoot "labels_ravdess_full.csv"
$MultiLabels = Join-Path $DataRoot "labels_audio_multi.csv"
$AgentAudioLabels = Join-Path $AgentRoot "audio\manifests\labels_audio_multi.csv"
$ManifestReport = Join-Path $ProjectRoot "results\multidataset_manifest_report.md"

function Invoke-LocalPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        if (($Arguments -contains "src\train_audio_wav2vec2.py") -and ($Arguments -contains "--device") -and ($Arguments -contains "cuda")) {
            Write-Host ""
            Write-Host "CUDA training failed. On a 4 GB GPU, Wav2Vec2 often runs out of memory." -ForegroundColor Yellow
            Write-Host "Retry safely on CPU:" -ForegroundColor Yellow
            Write-Host ".\run_local_audio_training.ps1 -Mode $Mode -Epochs $Epochs -Batch $Batch -LearningRate $LearningRate -UnfreezeLastN $UnfreezeLastN -Device cpu" -ForegroundColor Yellow
            Write-Host ""
        }
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Add-ExistingRoot {
    param(
        [System.Collections.Generic.List[string]]$Arguments,
        [string]$Root
    )
    if (Test-Path $Root) {
        $Arguments.Add("--root")
        $Arguments.Add($Root)
    }
}

if ($Mode -eq "manifest") {
    $arguments = [System.Collections.Generic.List[string]]::new()
    $arguments.Add("src\prepare_audio_emotion_manifest.py")
    if (Test-Path $RavdessLabels) {
        $arguments.Add("--manifest")
        $arguments.Add($RavdessLabels)
    }
    Add-ExistingRoot -Arguments $arguments -Root (Join-Path $ProjectRoot "data\raw\ALL")
    Add-ExistingRoot -Arguments $arguments -Root (Join-Path $DataRoot "raw\TESS")
    Add-ExistingRoot -Arguments $arguments -Root (Join-Path $DataRoot "raw\CREMA-D\AudioMP3")
    Add-ExistingRoot -Arguments $arguments -Root (Join-Path $DataRoot "raw\EmoDB")
    $arguments.Add("--output")
    $arguments.Add($MultiLabels)
    $arguments.Add("--report")
    $arguments.Add($ManifestReport)
    Invoke-LocalPython -Arguments $arguments.ToArray()
    exit 0
}

if ($Mode -eq "smoke") {
    $arguments = @(
        "src\train_audio_wav2vec2.py",
        "--labels", $RavdessLabels,
        "--output-dir", (Join-Path $DataRoot "models\wav2vec2_smoke"),
        "--epochs", "1",
        "--batch", "1",
        "--max-duration", "1.0",
        "--freeze-base",
        "--unfreeze-last-n", "0",
        "--limit-train", "2",
        "--limit-val", "2",
        "--limit-test", "2",
        "--no-save-model"
    )
    if ($Device -ne "auto") {
        $arguments += @("--device", $Device)
    }
    Invoke-LocalPython -Arguments $arguments
    exit 0
}

if ($Mode -eq "ravdess") {
    $labels = $RavdessLabels
    $outputDir = Join-Path $AgentRoot "audio\models\wav2vec2_ravdess"
}
else {
    $labels = if (Test-Path $AgentAudioLabels) { $AgentAudioLabels } else { $MultiLabels }
    $outputDir = Join-Path $AgentRoot "audio\models\wav2vec2_audio_multi"
}

$trainArgs = @(
    "src\train_audio_wav2vec2.py",
    "--labels", $labels,
    "--output-dir", $outputDir,
    "--epochs", [string]$Epochs,
    "--batch", [string]$Batch,
    "--lr", [string]$LearningRate,
    "--max-duration", [string]$MaxDuration,
    "--freeze-feature-encoder",
    "--freeze-base",
    "--unfreeze-last-n", [string]$UnfreezeLastN,
    "--gradient-accumulation-steps", [string]$GradientAccumulationSteps
)

if ($Device -ne "auto") {
    $trainArgs += @("--device", $Device)
}
if ($NoSaveModel) {
    $trainArgs += "--no-save-model"
}
if ($Amp) {
    $trainArgs += "--amp"
}

Invoke-LocalPython -Arguments $trainArgs
