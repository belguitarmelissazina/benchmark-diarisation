# Phase 4 - Speaker-count estimation ablation (2 methods x 4 files = 8 runs)
# Run from project root:  ./run_phase4.ps1

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

# Audio files + per-file oracle speaker count
$files = @(
    @{ name = "AMI-IS";  audio = "benchmarks/ami/IS1009a.wav";
       ref = "benchmarks/ami/IS1009a.rttm";                          transcribe = $false; nspk = 4 },
    @{ name = "AMI-EN";  audio = "benchmarks/ami/EN2002c.wav";
       ref = "benchmarks/ami/EN2002c.rttm";                          transcribe = $false; nspk = 3 },
    @{ name = "FR-018";  audio = "benchmarks/summre/018a_EARZ/018a_EARZ.wav";
       ref = "benchmarks/summre/018a_EARZ/018a_EARZ.ref.rttm";       transcribe = $true;  nspk = 4 },
    @{ name = "FR-069";  audio = "benchmarks/summre/069c_EEPL/069c_EEPL.wav";
       ref = "benchmarks/summre/069c_EEPL/069c_EEPL.ref.rttm";       transcribe = $true;  nspk = 4 }
)

# <<< EDIT after Phase 3 picks the winning embedding >>>
$BestEmbed = "resnet34"
$BestVad   = @("--vad-model", "silero", "--vad-threshold", "0.4")

$totalRuns = 2 * $files.Count
$runNum = 0

foreach ($f in $files) {
    # 4.1 NME-SC
    $runNum++
    $runName = "$($f.name)_nmesc"
    Write-Host ""
    Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan
    $cmd = @("-m", "diar_pipeline.run", "-i", $f.audio, "--reference-rttm", $f.ref,
             "--experiment", "ablate_estimate", "--run-name", $runName,
             "--embed", $BestEmbed, "--estimate", "nmesc") + $BestVad
    if ($f.transcribe) { $cmd += "--transcribe" }
    & $py @cmd

    # 4.2 Oracle (per-file speaker count)
    $runNum++
    $runName = "$($f.name)_oracle$($f.nspk)"
    Write-Host ""
    Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan
    $cmd = @("-m", "diar_pipeline.run", "-i", $f.audio, "--reference-rttm", $f.ref,
             "--experiment", "ablate_estimate", "--run-name", $runName,
             "--embed", $BestEmbed, "--num-speakers", "$($f.nspk)") + $BestVad
    if ($f.transcribe) { $cmd += "--transcribe" }
    & $py @cmd
}

Write-Host ""
Write-Host "=== Phase 4 complete ===" -ForegroundColor Green
Write-Host "If oracle is much better than nmesc, speaker counting is the bottleneck."
Write-Host "Proceed with nmesc as BEST_ESTIMATE for Phase 5."