# Phase: Re-test ecapa + campplus with fixed code (nmesc estimator)
# Old runs were broken: sim_enhancement bug + gmm_bic false single-speaker + nmesc always-k=1
# All three bugs are now fixed. Re-test with nmesc (best estimator so far).
# Run from project root:  ./run_phase_embed_v2.ps1

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

$BestVad = @("--vad-model", "silero", "--vad-threshold", "0.4")
$embeds  = @("ecapa", "campplus")

$totalRuns = $embeds.Count * $files.Count
$runNum = 0

foreach ($emb in $embeds) {
    foreach ($f in $files) {
        $runNum++
        $runName = "$($f.name)_${emb}_nmesc"
        Write-Host ""
        Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan

        $cmd = @(
            "-m", "diar_pipeline.run",
            "-i", $f.audio,
            "--reference-rttm", $f.ref,
            "--experiment", "ablate_embed_v2",
            "--run-name", $runName,
            "--embed", $emb,
            "--cluster", "sc",
            "--estimate", "nmesc"
        ) + $BestVad
        if ($f.transcribe) { $cmd += "--transcribe" }
        & $py @cmd
    }
}