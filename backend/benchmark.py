"""5개 STT 모델의 실시간 계수(RTF)를 이 노트북에서 실측하는 스크립트.

RTF = 처리 시간 / 오디오 길이. RTF < 1이면 실시간보다 빠르게 처리 가능,
RTF > 1이면 라이브 방송을 따라가지 못하고 지연이 계속 누적된다는 뜻이다.
"""

import time

import numpy as np

CHUNK_SECONDS = 4
SAMPLE_RATE = 16000


def make_test_chunk() -> bytes:
    """정확도 측정용이 아닌 순수 처리 속도 측정용 합성 오디오(사인파)."""
    t = np.linspace(0, CHUNK_SECONDS, CHUNK_SECONDS * SAMPLE_RATE, endpoint=False)
    tone = (0.1 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    return tone.tobytes()


def benchmark(name: str, build_adapter):
    print(f"\n[{name}] 모델 로딩 중...")
    load_start = time.perf_counter()
    adapter = build_adapter()
    load_time = time.perf_counter() - load_start

    chunk = make_test_chunk()
    infer_start = time.perf_counter()
    text = adapter.transcribe(chunk, SAMPLE_RATE)
    infer_time = time.perf_counter() - infer_start

    rtf = infer_time / CHUNK_SECONDS
    verdict = "실시간 가능" if rtf < 1 else "실시간 불가 (지연 누적)"
    print(f"  로딩 시간: {load_time:.2f}s")
    print(f"  청크({CHUNK_SECONDS}s) 처리 시간: {infer_time:.2f}s  →  RTF={rtf:.2f}  [{verdict}]")
    print(f"  출력(합성 사인파라 텍스트는 의미 없음): {text!r}")
    return rtf


def main():
    from adapters.whisper_adapter import FasterWhisperAdapter
    from adapters.sensevoice_adapter import SenseVoiceAdapter
    from adapters.moonshine_adapter import MoonshineAdapter

    results = {}
    results["faster-whisper (large-v3)"] = benchmark(
        "faster-whisper (large-v3)", lambda: FasterWhisperAdapter("large-v3")
    )
    results["faster-whisper (large-v3-turbo)"] = benchmark(
        "faster-whisper (large-v3-turbo)", lambda: FasterWhisperAdapter("large-v3-turbo")
    )
    results["faster-whisper (medium)"] = benchmark(
        "faster-whisper (medium)", lambda: FasterWhisperAdapter("medium")
    )
    results["SenseVoice"] = benchmark("SenseVoice", lambda: SenseVoiceAdapter())
    results["Moonshine-tiny-ko"] = benchmark("Moonshine-tiny-ko", lambda: MoonshineAdapter())

    print("\n=== 요약 ===")
    for name, rtf in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {name:35s} RTF={rtf:.2f}")


if __name__ == "__main__":
    main()
