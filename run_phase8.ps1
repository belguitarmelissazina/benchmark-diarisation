# Phase 8 - VBx refinement ablation (3 settings x 4 files = 12 runs)
# Run from project root:  ./run_phase8.ps1

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

$files = @(
    @{ name = "AMI-IS";  audio = "benchmarks/ami/IS1009a.wav";
       ref = "benchmarks/ami/IS1009a.rttm";                          transcribe = $false },
    @{ name = "AMI-EN";  audio = "benchmarks/ami/EN2002c.wav";
       ref = "benchmarks/ami/EN2002c.rttm";                          transcribe = $false },
    @{ name = "FR-018";  audio = "benchmarks/summre/018a_EARZ/018a_EARZ.wav";
       ref = "benchmarks/summre/018a_EARZ/018a_EARZ.ref.rttm";       transcribe = $true },
    @{ name = "FR-069";  audio = "benchmarks/summre/069c_EEPL/069c_EEPL.wav";
       ref = "benchmarks/summre/069c_EEPL/069c_EEPL.ref.rttm";       transcribe = $true }
)

# <<< EDIT after Phases 3-7 >>>
$BestEmbed    = "resnet34"
$BestEstimate = "gmm_bic"
$BestCluster  = "sc"
$EnhanceFlag  = @()
$BestWindow   = @("--win-len", "1.5", "--hop-len", "0.75")
$BestVad      = @("--vad-model", "silero", "--vad-threshold", "0.4")

$CommonFlags = @("--embed", $BestEmbed, "--estimate", $BestEstimate,
                 "--cluster", $BestCluster) + $BestWindow + $BestVad + $EnhanceFlag

$totalRuns = 3 * $files.Count
$runNum = 0

foreach ($f in $files) {
    # 8.1 No VBx
    $runNum++
    $runName = "$($f.name)_no_vbx"
    Write-Host ""
    Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan
    $cmd = @("-m", "diar_pipeline.run", "-i", $f.audio, "--reference-rttm", $f.ref,
             "--experiment", "ablate_vbx", "--run-name", $runName) + $CommonFlags
    if ($f.transcribe) { $cmd += "--transcribe" }
    & $py @cmd

    # 8.2 VBx default
    $runNum++
    $runName = "$($f.name)_vbx_default"
    Write-Host ""
    Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan
    $cmd = @("-m", "diar_pipeline.run", "-i", $f.audio, "--reference-rttm", $f.ref,
             "--experiment", "ablate_vbx", "--run-name", $runName,
             "--refine-vbx") + $CommonFlags
    if ($f.transcribe) { $cmd += "--transcribe" }
    & $py @cmd

    # 8.3 VBx aggressive
    $runNum++
    $runName = "$($f.name)_vbx_agg"
    Write-Host ""
    Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan
    $cmd = @("-m", "diar_pipeline.run", "-i", $f.audio, "--reference-rttm", $f.ref,
             "--experiment", "ablate_vbx", "--run-name", $runName,
             "--refine-vbx", "--vbx-Fa", "0.3", "--vbx-Fb", "25") + $CommonFlags
    if ($f.transcribe) { $cmd += "--transcribe" }
    & $py @cmd
}

Write-Host ""
Write-Host "=== Phase 8 complete ===" -ForegroundColor Green
Write-Host "Pick the lowest-DER VBx setting for Phase 9+."
