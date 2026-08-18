# Test the validated pipeline on dicte_audio_3.m4a (no ground-truth RTTM).
# Compares:
#   1) Diarize-then-transcribe  (diar_pipeline.run)
#   2) Transcribe-then-diarize  (diar_pipeline.run_transcribe_first)
# Pipeline config: resnet34 + nmesc + sc + silhouette_refine + silero VAD.

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), "Process")
        }
    }
    Write-Host "Loaded .env" -ForegroundColor DarkGray
}

$ErrorActionPreference = "Stop"
$py = "diarisation/Scripts/python.exe"
$audio = "dicte_audio_3.m4a"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "  [1/2] Diarize-then-transcribe" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

& $py -m diar_pipeline.run `
    -i $audio `
    --experiment "test_dicte3" `
    --run-name "dicte3_diar_then_trans" `
    --embed resnet34 `
    --cluster sc `
    --estimate nmesc `
    --silhouette-refine `
    --vad-model silero `
    --vad-threshold 0.4 `
    --transcribe

Write-Host ""
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "  [2/2] Transcribe-then-diarize" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

& $py -m diar_pipeline.run_transcribe_first `
    -i $audio `
    --experiment "test_dicte3" `
    --run-name "dicte3_trans_then_diar" `
    --embed resnet34 `
    --cluster sc `
    --estimate nmesc `
    --silhouette-refine `
    --vad-model silero `
    --vad-threshold 0.4

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Done. Check outputs in:" -ForegroundColor Green
Write-Host "    outputs/dicte_audio_3/         (diar_then_trans)" -ForegroundColor Green
Write-Host "    outputs/transcribe_first/...    (trans_then_diar)" -ForegroundColor Green
Write-Host "  Compare RTTMs + transcripts, or open in MLflow (experiment 'test_dicte3')." -ForegroundColor Green
