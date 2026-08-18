"""Download additional AMI test sessions (IHM Mix-Headset + SDM Array1-01).

Sessions: ES2004a, TS3003a, EN2002a, IS1009b, ES2004c
- IHM → benchmarks/ami/{session}.wav
- SDM → benchmarks/ami/sdm/{session}.wav
- RTTM refs from pyannote/AMI-diarization-setup (same for both)
"""
import os
import urllib.request
import sys

SESSIONS = ["ES2004a", "TS3003a", "EN2002a", "IS1009b", "ES2004c"]

AMI_AUDIO = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{s}/audio/{s}.{mic}.wav"
RTTM_URL = "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/only_words/rttms/test/{s}.rttm"

BASE = os.path.dirname(os.path.abspath(__file__))
IHM_DIR = os.path.join(BASE, "benchmarks", "ami")
SDM_DIR = os.path.join(BASE, "benchmarks", "ami", "sdm")
os.makedirs(SDM_DIR, exist_ok=True)


def download(url: str, dest: str):
    if os.path.exists(dest):
        print(f"  SKIP {os.path.basename(dest)} (exists)")
        return
    print(f"  GET  {os.path.basename(dest)} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"       -> {size_mb:.1f} MB")
    except Exception as e:
        print(f"  FAIL {e}")
        if os.path.exists(dest):
            os.remove(dest)


for s in SESSIONS:
    print(f"\n=== {s} ===")
    # IHM Mix-Headset
    download(AMI_AUDIO.format(s=s, mic="Mix-Headset"),
             os.path.join(IHM_DIR, f"{s}.wav"))
    # SDM Array1-01
    download(AMI_AUDIO.format(s=s, mic="Array1-01"),
             os.path.join(SDM_DIR, f"{s}.wav"))
    # RTTM (same reference for both)
    rttm_path = os.path.join(IHM_DIR, f"{s}.rttm")
    download(RTTM_URL.format(s=s), rttm_path)
    # Copy RTTM to SDM dir too
    sdm_rttm = os.path.join(SDM_DIR, f"{s}.rttm")
    if os.path.exists(rttm_path) and not os.path.exists(sdm_rttm):
        import shutil
        shutil.copy2(rttm_path, sdm_rttm)
        print(f"  COPY {s}.rttm -> sdm/")

print("\nDone.")
