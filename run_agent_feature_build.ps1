param(
    [ValidateSet("all", "original", "audio", "vision", "multimodal", "animation")]
    [string]$Mode = "all",

    [ValidateSet("cpu", "cuda", "auto")]
    [string]$Device = "cpu",

    [switch]$Overwrite
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
$RawDir = Join-Path $ProjectRoot "data\raw"

function Invoke-LocalPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Invoke-Precompute {
    param(
        [string]$Name,
        [string]$Labels,
        [string]$Output,
        [string[]]$ExtraArgs = @()
    )

    if (-not (Test-Path $Labels)) {
        Write-Host "Skipping $Name because manifest was not found: $Labels" -ForegroundColor Yellow
        return
    }

    if ((Test-Path $Output) -and (-not $Overwrite)) {
        Write-Host "Skipping $Name because feature cache already exists: $Output" -ForegroundColor Yellow
        Write-Host "Use -Overwrite to rebuild it." -ForegroundColor Yellow
        return
    }

    Write-Host ""
    Write-Host "Building $Name features -> $Output" -ForegroundColor Cyan
    $arguments = @(
        "src\precompute.py",
        "--labels", $Labels,
        "--raw-dir", $RawDir,
        "--output", $Output
    )
    if ($Device -ne "auto") {
        $arguments += @("--device", $Device)
    }
    if ($Overwrite) {
        $arguments += "--overwrite"
    }
    $arguments += $ExtraArgs
    Invoke-LocalPython -Arguments $arguments
}

Invoke-LocalPython -Arguments @("src\setup_agent_workspaces.py")

if ($Mode -eq "all" -or $Mode -eq "original") {
    Invoke-Precompute `
        -Name "Original SAVEE + FER fusion" `
        -Labels (Join-Path $ProjectRoot "data\labels.csv") `
        -Output (Join-Path $ProjectRoot "features\all_embeddings.pt")
}

if ($Mode -eq "all" -or $Mode -eq "audio") {
    Invoke-Precompute `
        -Name "Audio Agent multi-dataset" `
        -Labels (Join-Path $AgentRoot "audio\manifests\labels_audio_multi.csv") `
        -Output (Join-Path $AgentRoot "audio\features\audio_multi_embeddings.pt")
}

if ($Mode -eq "all" -or $Mode -eq "vision") {
    Invoke-Precompute `
        -Name "Vision Agent FER + RAVDESS frames" `
        -Labels (Join-Path $AgentRoot "vision\manifests\labels_vision.csv") `
        -Output (Join-Path $AgentRoot "vision\features\vision_embeddings.pt")
}

if ($Mode -eq "all" -or $Mode -eq "multimodal") {
    Invoke-Precompute `
        -Name "Fusion Agent RAVDESS audio-video" `
        -Labels (Join-Path $AgentRoot "multimodal\manifests\labels_ravdess_audio_video.csv") `
        -Output (Join-Path $AgentRoot "multimodal\features\ravdess_audio_video_embeddings.pt")
}

if ($Mode -eq "all" -or $Mode -eq "animation") {
    Invoke-Precompute `
        -Name "Animation Agent binary task" `
        -Labels (Join-Path $AgentRoot "animation\manifests\labels_animation.csv") `
        -Output (Join-Path $AgentRoot "animation\features\animation_embeddings.pt") `
        -ExtraArgs @("--class-names", "not_optimized,optimized")
}

Write-Host ""
Write-Host "Feature build finished for mode: $Mode" -ForegroundColor Green
