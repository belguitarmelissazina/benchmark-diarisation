# Phase: AHC + over-segmentation post-processing
# Goal: keep AHC's low confusion, fix the "too many speakers" problem.
# Two mechanisms:
#   --min-cluster-size N       drop clusters smaller than N embeddings
#   --merge-threshold D        iteratively merge clusters with centroid distance < D
# Run from project root:  ./run_phase_ahc_postproc.ps1

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

# Anchor on an AHC config that deliberately over-segments (low percentile)
# then let post-processing collapse it back to a sensible k.
$AHCPercentile = 30   # over-segments ~30 clusters on AMI-EN; post-proc collapses

# Post-processing configs to test
$configs = @(
    @{ label = "ms5";              args = @("--min-cluster-size", "5") },
    @{ label = "mt05";              args = @("--merge-threshold", "0.5") },
    @{ label = "mt07";              args = @("--merge-threshold", "0.7") },
    @{ label = "ms5_mt05";          args = @("--min-cluster-size", "5", "--merge-threshold", "0.5") },
    @{ label = "ms5_mt07";          args = @("--min-cluster-size", "5", "--merge-threshold", "0.7") }
)

$totalRuns = $configs.Count * $files.Count
$runNum = 0

Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "  AHC + post-processing: $totalRuns runs" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

foreach ($c in $configs) {
    foreach ($f in $files) {
        $runNum++
        $runName = "$($f.name)_ahc_p${AHCPercentile}_$($c.label)"
        Write-Host ""
        Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan

        $cmd = @(
            "-m", "diar_pipeline.run",
            "-i", $f.audio,
            "--reference-rttm", $f.ref,
            "--experiment", "ablate_ahc_postproc",
            "--run-name", $runName,
            "--embed", $BestEmbed,
            "--cluster", "ahc_threshold",
            "--ahc-percentile", "$AHCPercentile"
        ) + $BestVad + $c.args
        if ($f.transcribe) { $cmd += "--transcribe" }
        & $py @cmd
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Complete — compare ablate_ahc_postproc in MLflow" -ForegroundColor Green
Write-Host "  Filter: num_speakers_final close to ground truth (4,3,4,4)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
