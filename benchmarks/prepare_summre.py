"""Download and prepare SUMM-RE ASRU meetings for our diarization pipeline.

For each meeting, we:
  1. Download the per-speaker anonymized WAVs
  2. Download the per-speaker gold_ipus CSVs (speech/silence segments)
  3. Download the per-speaker gold transcript CSVs (TokensAlign tier)
  4. Mix the per-speaker WAVs into a single mono WAV via ffmpeg amix
  5. Build a reference RTTM by turning each 'ipu' row into a segment tagged
     with that speaker
  6. Build a reference transcript file by interleaving TokensAlign rows
     sorted by start time

Source: https://huggingface.co/datasets/datasets-CNRS/summ-re-asru (CC-BY-SA 4.0)

Usage:
  python -m benchmarks.prepare_summre
"""

from __future__ import annotations

import csv
import subprocess
import urllib.request
from pathlib import Path

import imageio_ffmpeg

BASE = "https://huggingface.co/datasets/datasets-CNRS/summ-re-asru/resolve/main"

# Two meetings, each with its list of speaker IDs
MEETINGS: dict[str, list[str]] = {
    "018a_EARZ": ["055", "056", "057", "058"],
    "020b_EBDZ": ["017", "053", "057", "063"],
    "027a_EBRH": ["025", "075", "078"],
    "032b_EADH": ["084", "085", "086", "087"],
    "033a_EBRH": ["091", "092", "093", "094"],
    "033c_EBPH": ["091", "092", "093", "094"],
    "034a_EBRH": ["095", "096", "097", "098"],
    "035b_EADH": ["088", "096", "097", "098"],
    "036c_EAPH": ["091", "092", "093", "099"],
    "069c_EEPL": ["156", "157", "158", "159"],
}

OUT_DIR = Path(__file__).resolve().parent / "summre"


# ── Download helpers ─────────────────────────────────────────────────────────

def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [get ] {dest.name}")
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def fetch_meeting(meeting: str, speakers: list[str], raw: Path) -> None:
    for spk in speakers:
        download(f"{BASE}/audio_anonymized/{meeting}_{spk}_anon.wav",
                 raw / f"{meeting}_{spk}.wav")
        download(f"{BASE}/gold_ipus/{meeting}_{spk}.csv",
                 raw / f"{meeting}_{spk}_ipus.csv")
        download(f"{BASE}/gold_transcriptions/csv/{meeting}_{spk}_anon.csv",
                 raw / f"{meeting}_{spk}_trans.csv")


# ── Mix per-speaker WAVs into a single mono WAV ──────────────────────────────

def mix_wavs(meeting: str, speakers: list[str], raw: Path, out_wav: Path) -> None:
    if out_wav.exists():
        print(f"  [skip] mixed {out_wav.name}")
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    inputs: list[str] = []
    for spk in speakers:
        inputs += ["-i", str(raw / f"{meeting}_{spk}.wav")]
    n = len(speakers)
    # amix normalizes by number of inputs -> each track at 1/n; bump back up
    filter_ = f"amix=inputs={n}:normalize=0,dynaudnorm=f=250:g=15"
    cmd = [ffmpeg, "-y", *inputs, "-filter_complex", filter_,
           "-ac", "1", "-ar", "16000", str(out_wav)]
    print(f"  [mix ] {out_wav.name} <- {n} tracks")
    subprocess.run(cmd, check=True, capture_output=True)


# ── Build reference RTTM from gold IPUs ──────────────────────────────────────

def build_rttm(meeting: str, speakers: list[str], raw: Path,
               out_rttm: Path) -> None:
    """Each speaker's ipus CSV has rows (tier, start, end, label).
    label == 'ipu' means that speaker is speaking. We emit one RTTM line
    per ipu, labeled with the speaker id."""
    segments: list[tuple[float, float, str]] = []
    for spk in speakers:
        csv_path = raw / f"{meeting}_{spk}_ipus.csv"
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4:
                    continue
                if row[3].strip().lower() != "ipu":
                    continue
                try:
                    s = float(row[1]); e = float(row[2])
                except ValueError:
                    continue
                if e > s:
                    segments.append((s, e, f"SPK{spk}"))
    segments.sort(key=lambda x: x[0])

    with open(out_rttm, "w", encoding="utf-8") as f:
        for s, e, spk in segments:
            dur = e - s
            f.write(f"SPEAKER {meeting} 1 {s:.3f} {dur:.3f} "
                    f"<NA> <NA> {spk} <NA> <NA>\n")
    print(f"  [rttm] {out_rttm.name} ({len(segments)} segments, "
          f"{len(set(s[2] for s in segments))} speakers)")


# ── Build reference transcript from gold TokensAlign rows ────────────────────

def build_transcript(meeting: str, speakers: list[str], raw: Path,
                     out_txt: Path, out_json: Path) -> None:
    """Merge the TokensAlign rows from every speaker into one time-sorted
    transcript. Silence tokens ('#') are dropped."""
    import json
    tokens: list[dict] = []
    for spk in speakers:
        csv_path = raw / f"{meeting}_{spk}_trans.csv"
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4:
                    continue
                if row[0].strip('"') != "TokensAlign":
                    continue
                tok = row[3].strip('"')
                if tok == "#" or not tok:
                    continue
                try:
                    s = float(row[1]); e = float(row[2])
                except ValueError:
                    continue
                tokens.append({"start": s, "end": e, "word": tok,
                               "speaker": f"SPK{spk}"})
    tokens.sort(key=lambda w: w["start"])

    # Group into turns (consecutive same-speaker tokens)
    turns: list[dict] = []
    for w in tokens:
        if turns and turns[-1]["speaker"] == w["speaker"]:
            turns[-1]["end"] = w["end"]
            turns[-1]["words"].append(w["word"])
        else:
            turns.append({"start": w["start"], "end": w["end"],
                          "speaker": w["speaker"], "words": [w["word"]]})
    for t in turns:
        t["text"] = " ".join(t.pop("words"))

    with open(out_txt, "w", encoding="utf-8") as f:
        for t in turns:
            sm, ss = divmod(t["start"], 60)
            em, es = divmod(t["end"], 60)
            f.write(f"[{int(sm):02d}:{ss:05.2f} - {int(em):02d}:{es:05.2f}]  "
                    f"{t['speaker']} : {t['text']}\n")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"words": tokens, "turns": turns}, f, ensure_ascii=False,
                  indent=2)
    print(f"  [trans] {out_txt.name} ({len(tokens)} words, {len(turns)} turns)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    for meeting, speakers in MEETINGS.items():
        print(f"\n=== {meeting} ({len(speakers)} speakers) ===")
        mdir = OUT_DIR / meeting
        raw = mdir / "raw"
        fetch_meeting(meeting, speakers, raw)
        mix_wavs(meeting, speakers, raw, mdir / f"{meeting}.wav")
        build_rttm(meeting, speakers, raw, mdir / f"{meeting}.ref.rttm")
        build_transcript(meeting, speakers, raw,
                         mdir / f"{meeting}.ref.txt",
                         mdir / f"{meeting}.ref.json")
    print("\nDone.")
    print(f"Artifacts in: {OUT_DIR}")


if __name__ == "__main__":
    main()
