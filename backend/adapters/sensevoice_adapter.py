import numpy as np
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

from adapters.base import STTModel


class SenseVoiceAdapter(STTModel):
    name = "SenseVoice"

    def __init__(self, device: str = "cpu"):
        self.model = AutoModel(model="iic/SenseVoiceSmall", trust_remote_code=True, device=device)

    def transcribe(self, audio_chunk: bytes, sample_rate: int) -> str:
        audio = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        result = self.model.generate(input=audio, language="ko", use_itn=True, fs=sample_rate)
        return rich_transcription_postprocess(result[0]["text"]).strip()
