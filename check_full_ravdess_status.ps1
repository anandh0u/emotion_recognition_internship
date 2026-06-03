$DataRoot = "E:\emotion_recognition_data"
$DownloadDir = Join-Path $DataRoot "downloads\ravdess"
$RawDir = Join-Path $DataRoot "raw\ravdess_full"
$ProcessedDir = Join-Path $DataRoot "processed\ravdess_full"
$FeaturesFile = Join-Path $DataRoot "features\ravdess_full_embeddings.pt"
$ModelsDir = Join-Path $DataRoot "models\ravdess_full"
$ResultsDir = Join-Path $DataRoot "results\ravdess_full"
$ErrLog = Join-Path $DataRoot "logs\full_ravdess_pipeline.err.log"
$OutLog = Join-Path $DataRoot "logs\full_ravdess_pipeline.out.log"

$processes = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like "*run_full_ravdess_e_drive.ps1*" -or
    $_.CommandLine -like "*download_ravdess_zenodo.py*" -or
    $_.CommandLine -like "*prepare_ravdess_manifest.py*" -or
    $_.CommandLine -like "*precompute.py*" -or
    $_.CommandLine -like "*train.py*" -or
    $_.CommandLine -like "*evaluate.py*"
}

$downloadedZips = @(Get-ChildItem $DownloadDir -File -Filter "Video_Speech_Actor_*.zip" -ErrorAction SilentlyContinue)
$partialZips = @(Get-ChildItem $DownloadDir -File -Filter "*.part" -ErrorAction SilentlyContinue)
$rawVideos = @(Get-ChildItem $RawDir -Recurse -File -Filter "*.mp4" -ErrorAction SilentlyContinue)
$frames = @(Get-ChildItem (Join-Path $ProcessedDir "frames") -File -Filter "*.jpg" -ErrorAction SilentlyContinue)
$audio = @(Get-ChildItem (Join-Path $ProcessedDir "audio") -File -Filter "*.wav" -ErrorAction SilentlyContinue)

[pscustomobject]@{
    RunningProcesses = $processes.Count
    DownloadedActorZips = $downloadedZips.Count
    PartialDownloads = $partialZips.Count
    RawVideoFiles = $rawVideos.Count
    ProcessedFrames = $frames.Count
    ProcessedAudioFiles = $audio.Count
    FeaturesReady = Test-Path $FeaturesFile
    BestModelReady = Test-Path (Join-Path $ModelsDir "best_model.pt")
    ResultsReady = Test-Path (Join-Path $ResultsDir "evaluation_all_metrics.json")
    EFreeGB = [math]::Round((Get-PSDrive E).Free / 1GB, 2)
}

if (Test-Path $ErrLog) {
    "`n--- latest pipeline log ---"
    Get-Content $ErrLog -Tail 12
}
if (Test-Path $OutLog) {
    "`n--- latest output log ---"
    Get-Content $OutLog -Tail 12
}
