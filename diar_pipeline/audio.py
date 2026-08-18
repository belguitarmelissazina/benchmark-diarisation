"""Audio helpers: convert to 16 kHz mono WAV + read duration."""

from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path


def convert_to_wav(src: Path, sr: int = 16000) -> Path:
    """Convert any audio file to 16 kHz mono WAV via ffmpeg (bundled)."""
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = Path(tempfile.gettempdir()) / f"{src.stem}_16k.wav"
    cmd = [
        ffmpeg_exe, "-y", "-v", "error", "-i", str(src),
        "-ac", "1", "-ar", str(sr), str(tmp),
    ]
    subprocess.run(cmd, check=True)
    return tmp


def get_audio_duration(audio_path: str | Path) -> float:
    """Return audio duration in seconds (soundfile, fallback torchaudio)."""
    import soundfile as sf
    try:
        return sf.info(str(audio_path)).duration
    except Exception:
        try:
            import torchaudio
            info = torchaudio.info(str(audio_path))
            return info.num_frames / info.sample_rate
        except Exception:
            return 0.0
