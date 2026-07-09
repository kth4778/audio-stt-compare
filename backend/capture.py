import asyncio
import subprocess
import sys
from pathlib import Path

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # s16le


class StreamCapture:
    """라이브 스트림을 청크(구간) 단위로 나눠 두 갈래로 내보낸다.

    1. STT용 원시 PCM (파이프, audio_chunks()로 소비)
    2. 구간별 리뷰용 영상 파일 (video_%05d.mp4, 음성 트랙 포함)

    영상은 -c:v copy(스트림 복사)로 자르면 가장 가까운 키프레임에서만 끊을 수 있어
    분석용 PCM 청크(정확히 chunk_seconds초 단위)와 시간 범위가 어긋난다 (실측 확인됨).
    그래서 -force_key_frames로 청크 경계마다 강제로 키프레임을 만들어 재인코딩하여,
    영상 구간과 분석 오디오 구간이 정확히 같은 시간 범위를 갖도록 한다.

    치지직/SOOP의 HLS를 ffmpeg가 직접 열면 "Invalid data found when
    processing input"로 실패한다 (CDN의 fMP4 세그먼트 처리 방식과 ffmpeg의 HLS
    디먹서가 맞지 않음 - 실측 확인됨). streamlink는 이 스트림을 문제없이 받아오므로,
    streamlink로 원본을 받아 ffmpeg에는 이미 정상화된 바이트 스트림만 파이프로 넘긴다.
    """

    def __init__(self, page_url: str, segment_dir: Path, chunk_seconds: int = 10):
        self.page_url = page_url
        self.segment_dir = segment_dir
        self.chunk_seconds = chunk_seconds
        self.chunk_bytes = chunk_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE
        self._streamlink_process: subprocess.Popen | None = None
        self._process: subprocess.Popen | None = None

    def start(self) -> None:
        self.segment_dir.mkdir(parents=True, exist_ok=True)

        streamlink_cmd = [sys.executable, "-m", "streamlink", self.page_url, "best", "-O", "--loglevel", "warning"]
        self._streamlink_process = subprocess.Popen(
            streamlink_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

        ffmpeg_cmd = [
            "ffmpeg",
            "-i", "pipe:0",
            "-map", "0:v", "-map", "0:a",
            "-c:v", "libx264", "-preset", "veryfast",
            "-force_key_frames", f"expr:gte(t,n_forced*{self.chunk_seconds})",
            "-c:a", "aac",
            "-f", "segment", "-segment_time", str(self.chunk_seconds),
            "-reset_timestamps", "1",
            "-segment_format", "mp4",
            "-segment_format_options", "movflags=frag_keyframe+empty_moov+default_base_moof",
            str(self.segment_dir / "video_%05d.mp4"),
            "-map", "0:a",
            "-ar", str(SAMPLE_RATE), "-ac", "1",
            "-f", "s16le", "pipe:1",
        ]
        self._process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=self._streamlink_process.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.chunk_bytes,
        )

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process = None
        if self._streamlink_process is not None:
            self._streamlink_process.terminate()
            self._streamlink_process = None

    async def audio_chunks(self):
        """(시퀀스 번호, PCM 오디오 바이트) 튜플을 생성하는 비동기 제너레이터."""
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("capture가 시작되지 않았습니다. start()를 먼저 호출하세요.")

        loop = asyncio.get_event_loop()
        seq = 0
        while True:
            chunk = await loop.run_in_executor(
                None, self._process.stdout.read, self.chunk_bytes
            )
            if not chunk:
                break
            yield seq, chunk
            seq += 1
