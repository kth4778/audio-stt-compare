from abc import ABC, abstractmethod


class STTModel(ABC):
    """청크 단위 오디오를 텍스트로 변환하는 STT 모델의 공통 인터페이스."""

    name: str

    @abstractmethod
    def transcribe(self, audio_chunk: bytes, sample_rate: int) -> str:
        """단일 오디오 청크(PCM 16-bit mono)를 받아 받아쓴 텍스트를 반환한다."""
        raise NotImplementedError
