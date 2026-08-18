# Phase 7 - Window / hop length ablation (3 windows x 4 files = 12 runs)
# Run from project root:  ./run_phase7.ps1

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

# <<< EDIT after Phases 3-5 (and 6 if applicable) >>>
$BestEmbed    = "resnet34"
$BestEstimate = "gmm_bic"
$BestCluster  = "sc"
$EnhanceFlag  = @()   # or: @("--no-enhance") if Phase 6 picked OFF
$BestVad      = @("--vad-model", "silero", "--vad-threshold", "0.4")

$windows = @(
    @{ tag = "win1.5"; len = "1.5"; hop = "0.75" },
    @{ tag = "win3.0"; len = "3.0"; hop = "1.5"  },
    @{ tag = "win5.0"; len = "5.0"; hop = "2.5"  }
)

$totalRuns = $windows.Count * $files.Count
$runNum = 0

foreach ($w in $windows) {
    foreach ($f in $files) {
        $runNum++
        $runName = "$($f.name)_$($w.tag)"
        Write-Host ""
        Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan

        $cmd = @("-m", "diar_pipeline.run", "-i", $f.audio, "--reference-rttm", $f.ref,
                 "--experiment", "ablate_window", "--run-name", $runName,
                 "--embed", $BestEmbed, "--estimate", $BestEstimate,
                 "--cluster", $BestCluster,
                 "--win-len", $w.len, "--hop-len", $w.hop) + $BestVad + $EnhanceFlag
        if ($f.transcribe) { $cmd += "--transcribe" }

        & $py @cmd
    }
}

Write-Host ""
Write-Host "=== Phase 7 complete ===" -ForegroundColor Green
Write-Host "Pick the lowest-DER window length and carry --win-len/--hop-len into Phase 8."
