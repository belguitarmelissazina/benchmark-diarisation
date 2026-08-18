# Phase 1 — Baselines (4 runs, ~25 min)
# Establishes default-config DER on every file.
# Run from project root:  ./run_phase1.ps1

$ErrorActionPreference = "Stop"
$py = "diarisation/Scripts/python.exe"

Write-Host "=== Phase 1.1 — AMI-IS baseline ===" -ForegroundColor Cyan
& $py -m diar_pipeline.run `
    -i benchmarks/ami/IS1009a.wav `
    --reference-rttm benchmarks/ami/IS1009a.rttm `
    --experiment baseline --run-name AMI-IS_baseline

Write-Host "=== Phase 1.2 — AMI-EN baseline ===" -ForegroundColor Cyan
& $py -m diar_pipeline.run `
    -i benchmarks/ami/EN2002c.wav `
    --reference-rttm benchmarks/ami/EN2002c.rttm `
    --experiment baseline --run-name AMI-EN_baseline

Write-Host "=== Phase 1.3 — FR-018 baseline (with transcription) ===" -ForegroundColor Cyan
& $py -m diar_pipeline.run `
    -i benchmarks/summre/018a_EARZ/018a_EARZ.wav `
    --reference-rttm benchmarks/summre/018a_EARZ/018a_EARZ.ref.rttm `
    --transcribe `
    --experiment baseline --run-name FR-018_baseline

Write-Host "=== Phase 1.4 — FR-069 baseline (with transcription) ===" -ForegroundColor Cyan
& $py -m diar_pipeline.run `
    -i benchmarks/summre/069c_EEPL/069c_EEPL.wav `
    --reference-rttm benchmarks/summre/069c_EEPL/069c_EEPL.ref.rttm `
    --transcribe `
    --experiment baseline --run-name FR-069_baseline

Write-Host ""
Write-Host "=== Phase 1 complete ===" -ForegroundColor Green
Write-Host "Open MLflow UI and note der_miss + der_false_alarm + der_confusion for AMI-IS."
Write-Host "If miss+FA > confusion, run ./run_phase2.ps1 next."
