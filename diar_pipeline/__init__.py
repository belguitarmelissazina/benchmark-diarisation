"""Standalone modular speaker diarization pipeline.

No dependency on the `diarize` PyPI package — everything is in-tree:
  - audio.py       : audio conversion + duration
  - models.py      : data structures (Segment, SpeechSegment, ...)
  - vad.py         : Silero VAD
  - embeddings.py  : WeSpeaker ResNet34-LM / CAM++ (ONNX)
  - clustering.py  : GMM-BIC, NME-SC, Spectral (+enhance), AHC, MeanShift
  - refinement.py  : VBx (Variational Bayes HMM)
  - segments.py    : assemble + write RTTM/TXT
  - run.py         : CLI orchestrator
"""

from .models import Segment, SpeechSegment, SubSegment, SpeakerEstimationDetails
from .audio import convert_to_wav, get_audio_duration
from .vad import run_vad
from .embeddings import extract_embeddings, EMBEDDING_WINDOW, EMBEDDING_STEP
from .clustering import (
    estimate_speakers_gmm_bic,
    estimate_speakers_nmesc,
    cluster_sc,
    cluster_ahc,
    cluster_meanshift,
    cluster_speakers,
    sim_enhancement,
)
from .refinement import refine_vbx
from .segments import build_segments, write_rttm, write_txt
from .transcription import transcribe, align_words_to_speakers, words_to_turns

__all__ = [
    "Segment", "SpeechSegment", "SubSegment", "SpeakerEstimationDetails",
    "convert_to_wav", "get_audio_duration",
    "run_vad",
    "extract_embeddings", "EMBEDDING_WINDOW", "EMBEDDING_STEP",
    "estimate_speakers_gmm_bic", "estimate_speakers_nmesc",
    "cluster_sc", "cluster_ahc", "cluster_meanshift", "cluster_speakers",
    "sim_enhancement",
    "refine_vbx",
    "build_segments", "write_rttm", "write_txt",
]
