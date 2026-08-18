# Phase: ablate_clustering + VBx refinement
# Same percentile sweep, but VBx should collapse over-segmentation to real k.
# Run from project root:  ./run_phase_clustering_vbx.ps1

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

$BestEmbed = "resnet34"
$BestVad   = @("--vad-model", "silero", "--vad-threshold", "0.4")

# AHC only — greedy didn't produce good DER in v2
$ahcPercentiles = @(20, 30)

$totalRuns = $ahcPercentiles.Count * $files.Count
$runNum = 0

Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "  Clustering + VBx: $totalRuns runs" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

foreach ($p in $ahcPercentiles) {
    foreach ($f in $files) {
        $runNum++
        $runName = "$($f.name)_ahc_p${p}_vbx"
        Write-Host ""
        Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan

        $cmd = @(
            "-m", "diar_pipeline.run",
            "-i", $f.audio,
            "--reference-rttm", $f.ref,
            "--experiment", "ablate_clustering_vbx",
            "--run-name", $runName,
            "--embed", $BestEmbed,
            "--cluster", "ahc_threshold",
            "--ahc-percentile", "$p",
            "--refine-vbx"
        ) + $BestVad
        if ($f.transcribe) { $cmd += "--transcribe" }
        & $py @cmd
    }
}

