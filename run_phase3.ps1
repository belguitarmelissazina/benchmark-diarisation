# Phase 3 - Embedding model ablation (3 embeds x 4 files = 12 runs)
# Run from project root:  ./run_phase3.ps1

# Load .env file (HF_TOKEN etc.)
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

# Audio files (same set as Phase 2)
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

# Winning VAD from Phase 2: silero @ th=0.4
$BestVad = @("--vad-model", "silero", "--vad-threshold", "0.4")

$embeds = @("resnet34", "campplus", "ecapa")

$totalRuns = $embeds.Count * $files.Count
$runNum = 0

foreach ($embed in $embeds) {
    foreach ($f in $files) {
        $runNum++
        $runName = "$($f.name)_$embed"
        Write-Host ""
        Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan

        $cmd = @(
            "-m", "diar_pipeline.run",
            "-i", $f.audio,
            "--reference-rttm", $f.ref,
            "--experiment", "ablate_embed",
            "--run-name", $runName,
            "--embed", $embed
        ) + $BestVad
        if ($f.transcribe) { $cmd += "--transcribe" }

        & $py @cmd
    }
}

Write-Host ""
Write-Host "=== Phase 3 complete ===" -ForegroundColor Green
Write-Host "In MLflow UI, pick the lowest-DER embedding (averaged across files) as BEST_EMBED."
