# Phase: Speaker-count estimation re-test after clustering.py fixes
#   - sim_enhancement now re-symmetrizes (fixed negative-eigenvalue bug)
#   - nmesc now uses absolute eigengap (fixed always-picks-k=1 bug)
#   - gmm_bic single-speaker pre-check now uses median>=0.75 (fixed false positives)
# Rerun on the same 4 files to get clean numbers.
# Run from project root:  ./run_phase_estimate_v2.ps1

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

# Skip runs already completed in this experiment
$doneRuns = @()
if (Test-Path "mlflow.db") {
    $doneRuns = & $py -c "import sqlite3; c=sqlite3.connect('mlflow.db'); cur=c.cursor(); cur.execute('SELECT r.name FROM runs r JOIN experiments e ON r.experiment_id=e.experiment_id WHERE e.name=''benchmark_final'' AND r.status=''FINISHED'' AND r.lifecycle_stage=''active'''); [print(r[0]) for r in cur.fetchall()]"
}

$files = @(
    @{ name = "AMI-EN2002a";  audio = "benchmarks/ami/EN2002a.wav";
       ref = "benchmarks/ami/EN2002a.rttm";                          transcribe = $false },
    @{ name = "AMI-ES2004a";  audio = "benchmarks/ami/ES2004a.wav";
       ref = "benchmarks/ami/ES2004a.rttm";                          transcribe = $false },
    @{ name = "AMI-ES2004c";  audio = "benchmarks/ami/ES2004c.wav";
       ref = "benchmarks/ami/ES2004c.rttm";                          transcribe = $false },
    @{ name = "AMI-IS1009b";  audio = "benchmarks/ami/IS1009b.wav";
       ref = "benchmarks/ami/IS1009b.rttm";                          transcribe = $false },
    @{ name = "AMI-TS3003a";  audio = "benchmarks/ami/TS3003a.wav";
       ref = "benchmarks/ami/TS3003a.rttm";                          transcribe = $false },
    @{ name = "SDM-EN2002a";  audio = "benchmarks/ami/sdm/EN2002a.wav";
       ref = "benchmarks/ami/sdm/EN2002a.rttm";                      transcribe = $false },
    @{ name = "SDM-ES2004a";  audio = "benchmarks/ami/sdm/ES2004a.wav";
       ref = "benchmarks/ami/sdm/ES2004a.rttm";                      transcribe = $false },
    @{ name = "SDM-ES2004c";  audio = "benchmarks/ami/sdm/ES2004c.wav";
       ref = "benchmarks/ami/sdm/ES2004c.rttm";                      transcribe = $false },
    @{ name = "SDM-IS1009b";  audio = "benchmarks/ami/sdm/IS1009b.wav";
       ref = "benchmarks/ami/sdm/IS1009b.rttm";                      transcribe = $false },
    @{ name = "SDM-TS3003a";  audio = "benchmarks/ami/sdm/TS3003a.wav";
       ref = "benchmarks/ami/sdm/TS3003a.rttm";                      transcribe = $false },
    @{ name = "FR-018";  audio = "benchmarks/summre/018a_EARZ/018a_EARZ.wav";
       ref = "benchmarks/summre/018a_EARZ/018a_EARZ.ref.rttm";       transcribe = $false },
    @{ name = "FR-020";  audio = "benchmarks/summre/020b_EBDZ/020b_EBDZ.wav";
       ref = "benchmarks/summre/020b_EBDZ/020b_EBDZ.ref.rttm";       transcribe = $false },
    @{ name = "FR-027";  audio = "benchmarks/summre/027a_EBRH/027a_EBRH.wav";
       ref = "benchmarks/summre/027a_EBRH/027a_EBRH.ref.rttm";       transcribe = $false },
    @{ name = "FR-032";  audio = "benchmarks/summre/032b_EADH/032b_EADH.wav";
       ref = "benchmarks/summre/032b_EADH/032b_EADH.ref.rttm";       transcribe = $false },
    @{ name = "FR-033a"; audio = "benchmarks/summre/033a_EBRH/033a_EBRH.wav";
       ref = "benchmarks/summre/033a_EBRH/033a_EBRH.ref.rttm";       transcribe = $false },
    @{ name = "FR-033c"; audio = "benchmarks/summre/033c_EBPH/033c_EBPH.wav";
       ref = "benchmarks/summre/033c_EBPH/033c_EBPH.ref.rttm";       transcribe = $false },
    @{ name = "FR-034";  audio = "benchmarks/summre/034a_EBRH/034a_EBRH.wav";
       ref = "benchmarks/summre/034a_EBRH/034a_EBRH.ref.rttm";       transcribe = $false },
    @{ name = "FR-035";  audio = "benchmarks/summre/035b_EADH/035b_EADH.wav";
       ref = "benchmarks/summre/035b_EADH/035b_EADH.ref.rttm";       transcribe = $false },
    @{ name = "FR-036";  audio = "benchmarks/summre/036c_EAPH/036c_EAPH.wav";
       ref = "benchmarks/summre/036c_EAPH/036c_EAPH.ref.rttm";       transcribe = $false },
    @{ name = "FR-069";  audio = "benchmarks/summre/069c_EEPL/069c_EEPL.wav";
       ref = "benchmarks/summre/069c_EEPL/069c_EEPL.ref.rttm";       transcribe = $false }
)

$BestEmbed = "resnet34"
$BestVad   = @("--vad-model", "silero", "--vad-threshold", "0.4")
$estimates = @("nmesc")

$totalRuns = $estimates.Count * $files.Count
$runNum = 0

Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "  Estimation re-test (post-fix): $totalRuns runs" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

foreach ($est in $estimates) {
    foreach ($f in $files) {
        $runNum++
        $runName = "$($f.name)_${est}_silref"
        if ($doneRuns -contains $runName) {
            Write-Host "=== [$runNum/$totalRuns] SKIP $runName (already done) ===" -ForegroundColor DarkGray
            continue
        }
        Write-Host ""
        Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan

        $cmd = @(
            "-m", "diar_pipeline.run",
            "-i", $f.audio,
            "--reference-rttm", $f.ref,
            "--experiment", "benchmark_final",
            "--run-name", $runName,
            "--embed", $BestEmbed,
            "--cluster", "sc",
            "--estimate", $est,
            "--silhouette-refine"
        ) + $BestVad
        if ($f.transcribe) { $cmd += "--transcribe" }
        & $py @cmd
    }
}
