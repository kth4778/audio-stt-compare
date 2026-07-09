import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Callable, List

from adapters.base import STTModel

OnResult = Callable[[int, str, str, float, dict | None], Awaitable[None]]


class Orchestrator:
    """오디오 청크 하나를 등록된 모든 STT 어댑터에 동시 디스패치하고,
    모델별로 완료되는 즉시 결과를 콜백으로 전달한다 (가장 느린 모델을 기다리지 않음)."""

    def __init__(self, adapters: List[STTModel], sample_rate: int, max_workers: int | None = None):
        self.adapters = adapters
        self.sample_rate = sample_rate
        self.executor = ThreadPoolExecutor(max_workers=max_workers or len(adapters))

    async def process_chunk(self, seq: int, audio_chunk: bytes, on_result: OnResult) -> None:
        loop = asyncio.get_event_loop()

        async def run_one(adapter: STTModel) -> None:
            start = time.perf_counter()
            text = await loop.run_in_executor(
                self.executor, adapter.transcribe, audio_chunk, self.sample_rate
            )
            latency = time.perf_counter() - start
            get_usage = getattr(adapter, "get_usage", None)
            usage = get_usage() if get_usage else None
            await on_result(seq, adapter.name, text, latency, usage)

        await asyncio.gather(*(run_one(adapter) for adapter in self.adapters))

    def shutdown(self) -> None:
        # cancel_futures=True: 아직 시작 안 한 대기 중인 작업(백로그)은 즉시 취소한다.
        # 이미 실행 중인 작업은 중간에 못 끊고 자연 종료된다 (몇 초 내 완료).
        self.executor.shutdown(wait=False, cancel_futures=True)
