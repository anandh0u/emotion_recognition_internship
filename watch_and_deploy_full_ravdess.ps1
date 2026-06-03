param(
    [string]$ProjectRoot = "E:\emotion_recognition_internship",
    [string]$DataRoot = "E:\emotion_recognition_data",
    [int]$PollSeconds = 300
)

$ErrorActionPreference = "Stop"

$FullModel = Join-Path $DataRoot "models\ravdess_full\best_model.pt"
$FullSummary = Join-Path $DataRoot "results\ravdess_full\training_summary.json"
$FullEvaluation = Join-Path $DataRoot "results\ravdess_full\evaluation_all_metrics.json"
$RepoModel = Join-Path $ProjectRoot "models\ravdess\best_model.pt"
$RepoResultsDir = Join-Path $ProjectRoot "results\ravdess"
$RepoSummary = Join-Path $RepoResultsDir "training_summary.json"
$RepoEvaluation = Join-Path $RepoResultsDir "evaluation_all_metrics.json"
$LogFile = Join-Path $DataRoot "logs\full_ravdess_deploy_watcher.log"

New-Item -ItemType Directory -Force -Path (Split-Path $LogFile), $RepoResultsDir | Out-Null

function Write-WatcherLog {
    param([string]$Message)
    $line = "$(Get-Date -Format s) $Message"
    Add-Content -Path $LogFile -Value $line
    Write-Output $line
}

Write-WatcherLog "Watcher started."

while (!(Test-Path $FullModel) -or !(Test-Path $FullSummary) -or !(Test-Path $FullEvaluation)) {
    Write-WatcherLog "Waiting for full RAVDESS outputs..."
    Start-Sleep -Seconds $PollSeconds
}

Write-WatcherLog "Full RAVDESS outputs found. Updating deployable app files."

Copy-Item -LiteralPath $FullModel -Destination $RepoModel -Force
Copy-Item -LiteralPath $FullSummary -Destination $RepoSummary -Force
Copy-Item -LiteralPath $FullEvaluation -Destination $RepoEvaluation -Force

Push-Location $ProjectRoot
try {
    git add -f models\ravdess\best_model.pt results\ravdess\training_summary.json results\ravdess\evaluation_all_metrics.json
    $status = git status --short
    if ($status) {
        git commit -m "Deploy full RAVDESS checkpoint"
        git push origin main
        Write-WatcherLog "Committed and pushed full RAVDESS checkpoint."
    }
    else {
        Write-WatcherLog "No deploy changes detected."
    }
}
finally {
    Pop-Location
}
