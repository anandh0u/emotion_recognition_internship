param(
    [ValidateSet("all", "original", "audio", "vision", "multimodal", "animation")]
    [string]$Mode = "all",

    [int]$Epochs = 30,
    [int]$Batch = 16,
    [double]$LearningRate = 1e-4,

    [ValidateSet("cpu", "cuda", "auto")]
    [string]$Device = "cpu"
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

function Invoke-LocalPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Invoke-FeatureTraining {
    param(
        [string]$Name,
        [string]$Features,
        [string]$ModelsDir,
        [string]$ResultsDir
    )

    if (-not (Test-Path $Features)) {
        Write-Host "Skipping $Name because feature cache was not found: $Features" -ForegroundColor Yellow
        Write-Host "Build it first with .\run_agent_feature_build.ps1" -ForegroundColor Yellow
        return
    }

    Write-Host ""
    Write-Host "Training $Name feature-head model" -ForegroundColor Cyan
    $arguments = @(
        "src\train.py",
        "--features", $Features,
        "--models-dir", $ModelsDir,
        "--results-dir", $ResultsDir,
        "--epochs", [string]$Epochs,
        "--batch", [string]$Batch,
        "--lr", [string]$LearningRate
    )
    if ($Device -ne "auto") {
        $arguments += @("--device", $Device)
    }
    Invoke-LocalPython -Arguments $arguments
}

if ($Mode -eq "all" -or $Mode -eq "original") {
    Invoke-FeatureTraining `
        -Name "Original SAVEE + FER fusion" `
        -Features (Join-Path $ProjectRoot "features\all_embeddings.pt") `
        -ModelsDir (Join-Path $AgentRoot "multimodal\models\savee_fer_fusion_head") `
        -ResultsDir (Join-Path $AgentRoot "multimodal\results\savee_fer_fusion_head")
}

if ($Mode -eq "all" -or $Mode -eq "audio") {
    Invoke-FeatureTraining `
        -Name "Audio Agent Wav2Vec2 embedding head" `
        -Features (Join-Path $AgentRoot "audio\features\audio_multi_embeddings.pt") `
        -ModelsDir (Join-Path $AgentRoot "audio\models\wav2vec2_embedding_head") `
        -ResultsDir (Join-Path $AgentRoot "audio\results\wav2vec2_embedding_head")
}

if ($Mode -eq "all" -or $Mode -eq "vision") {
    Invoke-FeatureTraining `
        -Name "Vision Agent ViT embedding head" `
        -Features (Join-Path $AgentRoot "vision\features\vision_embeddings.pt") `
        -ModelsDir (Join-Path $AgentRoot "vision\models\vit_embedding_head") `
        -ResultsDir (Join-Path $AgentRoot "vision\results\vit_embedding_head")
}

if ($Mode -eq "all" -or $Mode -eq "multimodal") {
    Invoke-FeatureTraining `
        -Name "Fusion Agent RAVDESS audio-video head" `
        -Features (Join-Path $AgentRoot "multimodal\features\ravdess_audio_video_embeddings.pt") `
        -ModelsDir (Join-Path $AgentRoot "multimodal\models\ravdess_fusion_head") `
        -ResultsDir (Join-Path $AgentRoot "multimodal\results\ravdess_fusion_head")
}

if ($Mode -eq "all" -or $Mode -eq "animation") {
    Invoke-FeatureTraining `
        -Name "Animation Agent binary head" `
        -Features (Join-Path $AgentRoot "animation\features\animation_embeddings.pt") `
        -ModelsDir (Join-Path $AgentRoot "animation\models\animation_head") `
        -ResultsDir (Join-Path $AgentRoot "animation\results\animation_head")
}

Write-Host ""
Write-Host "Feature-head training finished for mode: $Mode" -ForegroundColor Green
