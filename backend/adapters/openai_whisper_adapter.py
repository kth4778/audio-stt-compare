import io
import os
import wave

from openai import OpenAI

from adapters.base import STTModel


class OpenAIWhisperAdapter(STTModel):
    """OpenAI 오디오 전사 API 어댑터. 로컬 추론이 아니라 청크마다 HTTPS로
    오디오를 업로드해 결과를 받아온다 (과금 발생, OPENAI_API_KEY 필요).
    model 파라미터로 whisper-1 / gpt-4o-transcribe / gpt-4o-mini-transcribe
    중 하나를 골라 같은 /audio/transcriptions 엔드포인트를 쓴다."""

    # 분당 가격(USD) — whisper-1은 오래 안정적으로 알려진 값이라 확실하지만,
    # gpt-4o-transcribe/mini는 토큰 기반 과금이라 분당 환산치는 근사치다.
    # 정확한 단가는 반드시 platform.openai.com/usage에서 확인할 것.
    PRICE_PER_MINUTE_USD = {
        "whisper-1": 0.006,
        "gpt-4o-transcribe": 0.006,
        "gpt-4o-mini-transcribe": 0.003,
    }
    DEFAULT_PRICE_PER_MINUTE_USD = 0.006

    def __init__(self, model: str = "whisper-1", api_key: str | None = None):
        # 프론트엔드 모달에서 입력받은 키를 우선 사용하고, 없으면 서버 환경변수로 폴백한다.
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI API 키가 없습니다. 분석 시작 모달에서 API 키를 입력하거나 "
                "backend/.env 파일에 OPENAI_API_KEY=sk-... 를 추가하세요."
            )
        self.name = f"OpenAI API ({model})"
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.request_count = 0
        self.total_audio_seconds = 0.0

    def transcribe(self, audio_chunk: bytes, sample_rate: int) -> str:
        # API는 원시 PCM을 받지 않으므로 WAV 컨테이너로 감싸서 업로드한다.
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_chunk)
        buffer.seek(0)
        buffer.name = "chunk.wav"

        result = self.client.audio.transcriptions.create(
            model=self.model,
            file=buffer,
            language="ko",
        )
        self.request_count += 1
        self.total_audio_seconds += len(audio_chunk) / (sample_rate * 2)  # s16le = 2바이트/샘플
        return result.text.strip()

    def get_usage(self) -> dict:
        price = self.PRICE_PER_MINUTE_USD.get(self.model, self.DEFAULT_PRICE_PER_MINUTE_USD)
        return {
            "request_count": self.request_count,
            "total_audio_seconds": round(self.total_audio_seconds, 2),
            "estimated_cost_usd": round(self.total_audio_seconds / 60 * price, 4),
        }
