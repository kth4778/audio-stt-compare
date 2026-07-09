import numpy as np
from transformers import AutoProcessor, MoonshineForConditionalGeneration

from adapters.base import STTModel


class MoonshineAdapter(STTModel):
    name = "Moonshine-tiny-ko"

    def __init__(self, device: str = "cpu"):
        model_id = "UsefulSensors/moonshine-tiny-ko"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = MoonshineForConditionalGeneration.from_pretrained(model_id).to(device)
        self.device = device

    def transcribe(self, audio_chunk: bytes, sample_rate: int) -> str:
        audio = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs)
        return self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()
