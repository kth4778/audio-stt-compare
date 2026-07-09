import numpy as np
from faster_whisper import WhisperModel

from adapters.base import STTModel


class FasterWhisperAdapter(STTModel):
    """faster-whisper 기반 어댑터. size로 large-v3 / large-v3-turbo / medium 등을 지정한다."""

    def __init__(self, size: str, device: str = "cpu", compute_type: str = "int8"):
        self.name = f"faster-whisper ({size})"
        self.model = WhisperModel(size, device=device, compute_type=compute_type)

    def transcribe(self, audio_chunk: bytes, sample_rate: int) -> str:
        audio = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(audio, language="ko", vad_filter=True)
        return "".join(segment.text for segment in segments).strip()
