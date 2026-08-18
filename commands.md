# Diarization + Transcription — Benchmark Plan

diarisation/Scripts/python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000 --workers 1


All commands use:
- Working directory: project root
- Python: `diarisation/Scripts/python.exe` (replace with `python` if venv is already active)
- Tracking store: `sqlite:///mlflow.db` (metadata) + `./mlruns` (artifacts). The SQLite DB is required for the MLflow UI Overview tab.

Every run logs params, per-step metrics, all artifacts, and DER (when `--reference-rttm` is given).

**Files used:**
| label | file | spk | dur | role |
|---|---|---|---|---|
| **AMI-IS** | `benchmarks/ami/IS1009a.wav` | 4 | ~13 min | English, fast → main ablation file |
| **AMI-EN** | `benchmarks/ami/EN2002c.wav` | 3 | ~48 min | English, long → robustness check |
| **FR-018** | `benchmarks/summre/018a_EARZ/018a_EARZ.wav` | 4 | ~20 min | French, full pipeline (DER + ASR) |
| **FR-069** | `benchmarks/summre/069c_EEPL/069c_EEPL.wav` | 4 | ~20 min | French, full pipeline (DER + ASR) |

**Strategy:** ablate one knob at a time on **AMI-IS** (shortest file, fast iteration), pick the best config, validate it on **AMI-EN**, then run the full pipeline on **FR-018** and **FR-069** including transcription.

---

## Phase 0 — Setup (one-time)

### 0a. Prepare SUMM-RE data

```bash
diarisation/Scripts/python.exe -m benchmarks.prepare_summre
```

### 0b. (optional) Start MLflow UI in another terminal — for cross-run comparison

```bash
diarisation/Scripts/python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
```

### 0c. (optional) Start custom dashboard for single-run inspection

```bash
# Terminal 1
diarisation/Scripts/python.exe -m uvicorn backend.app:app --port 8765
# Terminal 2
cd frontend && npm run dev
```

---

## Phase 1 — Baselines (4 runs)

Establish a default-config DER number on every file. **Run all four before doing anything else.** These tell you whether the pipeline works at all on each file and give you reference values to beat.

### 1.1 AMI-IS baseline

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/IS1009a.wav --reference-rttm benchmarks/ami/IS1009a.rttm --experiment baseline --run-name AMI-IS_baseline
```

### 1.2 AMI-EN baseline

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/EN2002c.wav --reference-rttm benchmarks/ami/EN2002c.rttm --experiment baseline --run-name AMI-EN_baseline
```

### 1.3 FR-018 baseline (with transcription)

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/summre/018a_EARZ/018a_EARZ.wav --reference-rttm benchmarks/summre/018a_EARZ/018a_EARZ.ref.rttm --transcribe --experiment baseline --run-name FR-018_baseline
```

### 1.4 FR-069 baseline (with transcription)

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/summre/069c_EEPL/069c_EEPL.wav --reference-rttm benchmarks/summre/069c_EEPL/069c_EEPL.ref.rttm --transcribe --experiment baseline --run-name FR-069_baseline
```

**After Phase 1:** open MLflow UI, sort `baseline` experiment by DER. Note the baseline DER for AMI-IS — you'll compare every Phase-2..9 run against it. **Also write down the `der_miss`, `der_false_alarm`, and `der_confusion` values separately** — you need them to decide which phase to prioritize (see "How to read DER components" below).

---

## Phase 2 — VAD ablation: threshold sweep + model comparison (28 runs, ~2h30)

**Run this BEFORE embedding ablation.** Phase 1 showed that VAD error is the dominant DER
component on AMI-IS (FA-heavy) and FR-018 (miss-heavy), but each file needs a different
direction of tuning. This phase tests:

- **Phase 2A** — Silero threshold sweep: 6 thresholds × 4 files = 24 runs
- **Phase 2B** — pyannote segmentation-3.0 as an alternative VAD: 4 runs

All variants use the default embedding/cluster — only VAD changes.

### Phase 2A — Silero threshold sweep

| threshold | expected effect |
|---|---|
| 0.30 | very permissive — rescues missed speech on French (high-miss files) |
| 0.35 | moderately permissive |
| 0.40 | just below default |
| 0.45 | baseline (for comparison) |
| 0.55 | moderately strict — should cut FA on AMI |
| 0.65 | very strict — aggressive FA reduction |

Each threshold runs on all 4 files (AMI-IS, AMI-EN, FR-018, FR-069) to produce
a per-file threshold curve. The script `run_phase2.ps1` loops automatically.

### Phase 2B — pyannote segmentation-3.0

A fundamentally different VAD model trained on meeting audio. Uses the pyannote
`segmentation-3.0` model with `VoiceActivityDetection` pipeline wrapper. Requires:

```bash
pip install pyannote.audio
$env:HF_TOKEN = "hf_xxxxx"  # https://huggingface.co/settings/tokens
# Accept license at https://huggingface.co/pyannote/segmentation-3.0
```

Runs on all 4 files with default onset=0.50.

### Running Phase 2

```powershell
./run_phase2.ps1
```

The script runs all 28 variants sequentially (24 Silero + 4 pyannote). If `HF_TOKEN`
is not set, pyannote runs are skipped with a warning.

### How to read the results

In MLflow UI → `ablate_vad` experiment:
1. Add columns: `vad_model`, `vad_threshold`, `der`, `der_miss`, `der_false_alarm`
2. Group by file (run-name prefix: AMI-IS, AMI-EN, FR-018, FR-069)
3. For each file, plot `vad_threshold` (x-axis) vs `der_miss` / `der_false_alarm` (y-axis)
4. The optimal threshold is different per file — that's expected (English vs French)

**Decision rule:**
- If pyannote beats all Silero variants on most files → `BEST_VAD = --vad-model pyannote`
- If a single Silero threshold wins across all files → `BEST_VAD = --vad-threshold <winner>`
- If the optimal threshold diverges by language → use per-language configs in Phases 10-11

Pick `BEST_VAD` and carry its flags into Phases 3-8.

---

## Phase 3 — Ablate the embedding model (3 runs)

Biggest single knob after VAD. Different speaker embeddings have very different separability characteristics. Runs reuse `BEST_VAD` flags (e.g. `--vad-threshold 0.55 --vad-min-speech-ms 400 --vad-pad-ms 10`).

### 3.1 ResNet34-LM (256d, WeSpeaker)

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/IS1009a.wav --reference-rttm benchmarks/ami/IS1009a.rttm --experiment ablate_embed --run-name AMI-IS_resnet34 --embed resnet34 [BEST_VAD flags]
```

### 3.2 CAM++ (512d, WeSpeaker)

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/IS1009a.wav --reference-rttm benchmarks/ami/IS1009a.rttm --experiment ablate_embed --run-name AMI-IS_campplus --embed campplus [BEST_VAD flags]
```

### 3.3 ECAPA-TDNN (192d, SpeechBrain)

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/IS1009a.wav --reference-rttm benchmarks/ami/IS1009a.rttm --experiment ablate_embed --run-name AMI-IS_ecapa --embed ecapa [BEST_VAD flags]
```

**Decision:** keep the lowest-DER embedding as `BEST_EMBED` for Phases 4-8.

---

## Phase 4 — Ablate speaker-count estimation (3 runs)

Wrong speaker count is the most common DER killer. Test both estimators.

### 4.1 GMM + BIC

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/IS1009a.wav --reference-rttm benchmarks/ami/IS1009a.rttm --experiment ablate_estimate --run-name AMI-IS_gmm_bic --embed BEST_EMBED --estimate gmm_bic [BEST_VAD flags]
```

### 4.2 NME-SC (Normalized Maximum Eigengap)

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/IS1009a.wav --reference-rttm benchmarks/ami/IS1009a.rttm --experiment ablate_estimate --run-name AMI-IS_nmesc --embed BEST_EMBED --estimate nmesc [BEST_VAD flags]
```

### 4.3 Oracle (cheating — force the true count to see the floor)

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/IS1009a.wav --reference-rttm benchmarks/ami/IS1009a.rttm --experiment ablate_estimate --run-name AMI-IS_oracle4 --embed BEST_EMBED --num-speakers 4 [BEST_VAD flags]
```

**Reading the result:** if `oracle4` is much better than `gmm_bic`/`nmesc`, the bottleneck is speaker counting, not the rest of the pipeline. Pick `BEST_ESTIMATE` for Phases 5-8.

---

## Phase 5 — Ablate clustering algorithm (3 runs)

### 5.1 Spectral clustering (default)

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/IS1009a.wav --reference-rttm benchmarks/ami/IS1009a.rttm --experiment ablate_cluster --run-name AMI-IS_sc --embed BEST_EMBED --estimate BEST_ESTIMATE --cluster sc
```

### 5.2 Agglomerative hierarchical

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/IS1009a.wav --reference-rttm benchmarks/ami/IS1009a.rttm --experiment ablate_cluster --run-name AMI-IS_ahc --embed BEST_EMBED --estimate BEST_ESTIMATE --cluster ahc
```

### 5.3 Mean shift (no fixed k)

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/IS1009a.wav --reference-rttm benchmarks/ami/IS1009a.rttm --experiment ablate_cluster --run-name AMI-IS_meanshift --embed BEST_EMBED --estimate BEST_ESTIMATE --cluster meanshift
```

**Pick `BEST_CLUSTER` for the rest.** Mean shift ignores `--num-speakers` — useful as a sanity check against the estimator.

---

## Phase 6 — Ablate similarity enhancement (2 runs, SC only)

Only meaningful when `BEST_CLUSTER == sc`. The enhancement is: diagonal fill → gaussian blur → row threshold → symmetrize → diffusion → row max norm.

### 6.1 Enhancement ON (default)

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/IS1009a.wav --reference-rttm benchmarks/ami/IS1009a.rttm --experiment ablate_enhance --run-name AMI-IS_enhance_on --embed BEST_EMBED --estimate BEST_ESTIMATE --cluster sc
```

### 6.2 Enhancement OFF

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/IS1009a.wav --reference-rttm benchmarks/ami/IS1009a.rttm --experiment ablate_enhance --run-name AMI-IS_enhance_off --embed BEST_EMBED --estimate BEST_ESTIMATE --cluster sc --no-enhance
```

---

## Phase 7 — Ablate window / hop length (3 runs)

Embedding sliding window length. Short = more segments, more localized but noisier embeddings. Long = stable embeddings but worse boundaries.

### 7.1 Short window

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/IS1009a.wav --reference-rttm benchmarks/ami/IS1009a.rttm --experiment ablate_window --run-name AMI-IS_win1.5 --embed BEST_EMBED --estimate BEST_ESTIMATE --cluster BEST_CLUSTER --win-len 1.5 --hop-len 0.75
```

### 7.2 Medium window (default)

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/IS1009a.wav --reference-rttm benchmarks/ami/IS1009a.rttm --experiment ablate_window --run-name AMI-IS_win3.0 --embed BEST_EMBED --estimate BEST_ESTIMATE --cluster BEST_CLUSTER --win-len 3.0 --hop-len 1.5
```

### 7.3 Long window

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/IS1009a.wav --reference-rttm benchmarks/ami/IS1009a.rttm --experiment ablate_window --run-name AMI-IS_win5.0 --embed BEST_EMBED --estimate BEST_ESTIMATE --cluster BEST_CLUSTER --win-len 5.0 --hop-len 2.5
```

---

## Phase 8 — VBx refinement (3 runs)

VBx is a variational-Bayes HMM smoother that re-assigns embeddings to clusters. It usually helps but adds time.

### 8.1 No VBx

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/IS1009a.wav --reference-rttm benchmarks/ami/IS1009a.rttm --experiment ablate_vbx --run-name AMI-IS_no_vbx --embed BEST_EMBED --estimate BEST_ESTIMATE --cluster BEST_CLUSTER
```

### 8.2 VBx default (Fa=0.4, Fb=17)

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/IS1009a.wav --reference-rttm benchmarks/ami/IS1009a.rttm --experiment ablate_vbx --run-name AMI-IS_vbx_default --embed BEST_EMBED --estimate BEST_ESTIMATE --cluster BEST_CLUSTER --refine-vbx
```

### 8.3 VBx aggressive (lower Fa, higher Fb = stronger smoothing)

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/IS1009a.wav --reference-rttm benchmarks/ami/IS1009a.rttm --experiment ablate_vbx --run-name AMI-IS_vbx_agg --embed BEST_EMBED --estimate BEST_ESTIMATE --cluster BEST_CLUSTER --refine-vbx --vbx-Fa 0.3 --vbx-Fb 25
```

---

## Phase 9 — Silhouette refinement (1 run, optional)

Adds a post-cluster cleanup pass that removes low-confidence assignments.

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/IS1009a.wav --reference-rttm benchmarks/ami/IS1009a.rttm --experiment ablate_silhouette --run-name AMI-IS_silhouette --embed BEST_EMBED --estimate BEST_ESTIMATE --cluster BEST_CLUSTER --silhouette-refine
```

**After Phase 9:** in MLflow UI, sort all `ablate_*` experiments by DER. Pick the winning combination → call this `BEST_CONFIG`. Replace `BEST_*` placeholders below with the chosen flag values.

---

## Phase 10 — Validate BEST_CONFIG on AMI-EN (1 run)

The long-file robustness check. Same config that won on AMI-IS (13 min) must still beat baseline on AMI-EN (48 min).

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/EN2002c.wav --reference-rttm benchmarks/ami/EN2002c.rttm --experiment validate --run-name AMI-EN_best [BEST_CONFIG flags]
```

If DER on AMI-EN is **worse** than its baseline, the winner from Phase 1-9 was overfit to the short file. Falling back to default is fine.

---

## Phase 11 — French full pipeline with BEST_CONFIG (2 runs)

Diarization + transcription + DER on both SUMM-RE meetings.

### 11.1 FR-018

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/summre/018a_EARZ/018a_EARZ.wav --reference-rttm benchmarks/summre/018a_EARZ/018a_EARZ.ref.rttm --transcribe --experiment french_final --run-name FR-018_best [BEST_CONFIG flags]
```

### 11.2 FR-069

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/summre/069c_EEPL/069c_EEPL.wav --reference-rttm benchmarks/summre/069c_EEPL/069c_EEPL.ref.rttm --transcribe --experiment french_final --run-name FR-069_best [BEST_CONFIG flags]
```

**Compare** the resulting transcript against the gold transcript at:
- `benchmarks/summre/018a_EARZ/018a_EARZ.ref.txt`
- `benchmarks/summre/069c_EEPL/069c_EEPL.ref.txt`

These are the human-annotated verbatim French transcripts shipped by SUMM-RE-ASRU.

---

## Phase 12 — Oracle / sanity bounds (3 runs, optional)

Useful for understanding *why* a number is what it is.

### 12.1 Oracle speaker count on FR-018

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/summre/018a_EARZ/018a_EARZ.wav --reference-rttm benchmarks/summre/018a_EARZ/018a_EARZ.ref.rttm --transcribe --experiment oracle --run-name FR-018_oracle4 [BEST_CONFIG flags] --num-speakers 4
```

### 12.2 Oracle speaker count on FR-069

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/summre/069c_EEPL/069c_EEPL.wav --reference-rttm benchmarks/summre/069c_EEPL/069c_EEPL.ref.rttm --transcribe --experiment oracle --run-name FR-069_oracle4 [BEST_CONFIG flags] --num-speakers 4
```

### 12.3 Oracle on AMI-EN (3 speakers)

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/ami/EN2002c.wav --reference-rttm benchmarks/ami/EN2002c.rttm --experiment oracle --run-name AMI-EN_oracle3 [BEST_CONFIG flags] --num-speakers 3
```

**Reading:** `oracle - best` gap = how much DER comes from speaker-count errors vs. clustering errors.

---

## Phase 13 — Min/max speaker range (1 run, optional)

If oracle is far better than estimation, try giving the estimator a tight prior instead of a hard count.

```bash
diarisation/Scripts/python.exe -m diar_pipeline.run -i benchmarks/summre/018a_EARZ/018a_EARZ.wav --reference-rttm benchmarks/summre/018a_EARZ/018a_EARZ.ref.rttm --transcribe --experiment oracle --run-name FR-018_range3-5 [BEST_CONFIG flags] --min-speakers 3 --max-speakers 5
```

---

## Total run count

| phase | runs | rough wall-clock |
|---|---|---|
| 1 — baselines | 4 | ~25 min |
| 2 — VAD | 28 | ~2h30 |
| 3 — embed | 3 | ~9 min |
| 4 — estimate | 3 | ~9 min |
| 5 — cluster | 3 | ~9 min |
| 6 — enhance | 2 | ~6 min |
| 7 — window | 3 | ~9 min |
| 8 — VBx | 3 | ~9 min |
| 9 — silhouette | 1 | ~3 min |
| 10 — validate | 1 | ~10 min |
| 11 — French final | 2 | ~12 min |
| 12 — oracle | 3 | ~15 min |
| 13 — range | 1 | ~6 min |
| **total** | **57** | **~5h** |

(Phases 2-9 use ~3 min each because AMI-IS is 13 min audio. AMI-EN is ~10 min/run because 48 min audio.)

---

## How to read the results

1. **MLflow UI** (`http://localhost:5000`): cross-run comparison. Click on an experiment, sort by `der_total` ascending. Use "Compare" to see parallel coordinates of params vs DER.
2. **Custom dashboard** (`http://localhost:3000`): single-run inspection — timeline, waveform, UMAP, similarity, transcript with click-to-play.
3. **Local files** in `mlruns/` survive frontend restarts. Nothing is lost.

### How to read DER components

**DER = miss + false_alarm + confusion.** Each component points at a different part of the pipeline:

| component | meaning | blame |
|---|---|---|
| `der_miss` | % of reference speech labeled as silence | **VAD** — missed real speech |
| `der_false_alarm` | % of reference silence labeled as speech | **VAD** — flagged noise as speech |
| `der_confusion` | % of speech with wrong speaker label | **embeddings + clustering** |

**Attribution rule:**
- `VAD error = der_miss + der_false_alarm`
- `Clustering error = der_confusion`
- Whichever is larger is your bottleneck. Tune that first.

**Miss/FA ratio diagnoses WHICH way to tune VAD:**
- `FA > miss` → VAD too permissive → raise `--vad-threshold`, drop `--vad-pad-ms`, raise `--vad-min-speech-ms`
- `miss > FA` → VAD too strict → lower `--vad-threshold`, add padding
- `miss ≈ FA` → VAD balanced; little room left for single-knob tuning

**Example (AMI-IS baseline):** `DER=10.68%, miss=2.62%, FA=4.84%, confusion=3.22%`. VAD error (7.46%) > clustering error (3.22%) → fix VAD first. FA > miss → raise threshold. That's how Phase 2 came to exist.

### Key metrics to track

| metric | meaning |
|---|---|
| `der_total` | overall diarization error rate (%) — lower better |
| `der_miss` | missed speech % |
| `der_false_alarm` | non-speech labeled as speech % |
| `der_confusion` | wrong speaker assigned % |
| `n_speakers_estimated` vs `n_speakers_reference` | speaker-count accuracy |
| `time_*` | per-step latency |
| `ram_peak_mb` | peak RSS during run |

### Comparing transcripts (French only)

Reference at `benchmarks/summre/{meeting}/{meeting}.ref.txt`. Pipeline output is logged to MLflow as `transcript/transcript.txt`. Open both side-by-side, or write a small WER script if you want a number — currently we only compute DER, not WER (per your earlier instruction).

---

## Resetting / cleaning up

```bash
# Delete a single run: use MLflow UI delete button
# Delete an experiment: MLflow UI

# Nuke everything (CAREFUL — irreversible)
rm -rf mlruns/
```
