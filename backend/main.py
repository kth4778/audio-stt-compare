import asyncio
import json
import logging
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI

import db
from adapters.openai_whisper_adapter import OpenAIWhisperAdapter
from capture import SAMPLE_RATE, StreamCapture
from orchestrator import Orchestrator

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt-compare")

SEGMENT_DIR = Path(__file__).parent / "segments"

# Render 무료 플랜처럼 RAM이 빠듯한 클라우드 배포에서는 ffmpeg의 영상 재인코딩(libx264)이
# 메모리를 초과시켜 컨테이너가 죽는 문제가 있어(실측 확인됨), 배포 환경에서는
# CAPTURE_VIDEO=false로 두고 오디오만 캡처한다. 로컬 개발은 기본값 그대로 영상 유지.
CAPTURE_VIDEO = os.environ.get("CAPTURE_VIDEO", "true").lower() != "false"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    """세션별로 WebSocket 연결을 나눠 관리한다 — 여러 명이 동시에 서로 다른
    세션을 분석해도 각자 자기 세션의 결과만 받아야 하기 때문."""

    def __init__(self):
        self.active: dict[str, list[WebSocket]] = {}

    async def connect(self, session_name: str, ws: WebSocket):
        await ws.accept()
        self.active.setdefault(session_name, []).append(ws)

    def disconnect(self, session_name: str, ws: WebSocket):
        conns = self.active.get(session_name)
        if conns and ws in conns:
            conns.remove(ws)
            if not conns:
                del self.active[session_name]

    async def broadcast(self, session_name: str, message: dict):
        for ws in list(self.active.get(session_name, [])):
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(session_name, ws)


manager = ConnectionManager()

# 세션명 -> {"capture", "orchestrator", "last_seq", "task"} — 동시에 여러 명이
# 각자 다른(또는 같은) 방송을 분석할 수 있도록 세션마다 독립적으로 상태를 갖는다.
# 인원 제한은 두지 않는다 (CPU 부담이 큰 로컬 모델을 여럿이 동시에 고르면
# 당연히 서로 느려지지만, 이건 사용자들이 알아서 모델을 고르며 관리할 몫이다).
sessions: dict[str, dict] = {}


def _faster_whisper(size: str):
    # torch/faster-whisper는 무거워서(수 GB) 클라우드 배포 이미지에는 안 넣으므로,
    # 실제로 이 모델을 고를 때만 임포트한다 — 로컬 개발 환경에는 그대로 설치돼 있어
    # 정상 동작하고, 가벼운 클라우드 배포본에서 고르면 명확한 에러로 안내된다.
    from adapters.whisper_adapter import FasterWhisperAdapter

    return FasterWhisperAdapter(size)


def _sensevoice():
    from adapters.sensevoice_adapter import SenseVoiceAdapter

    return SenseVoiceAdapter()


def _moonshine():
    from adapters.moonshine_adapter import MoonshineAdapter

    return MoonshineAdapter()


MODEL_REGISTRY = {
    "large-v3": lambda: _faster_whisper("large-v3"),
    "large-v3-turbo": lambda: _faster_whisper("large-v3-turbo"),
    "medium": lambda: _faster_whisper("medium"),
    "sensevoice": _sensevoice,
    "moonshine": _moonshine,
}

OPENAI_API_MODEL_KEYS = {
    "openai-whisper-1": "whisper-1",
    "openai-gpt-4o-transcribe": "gpt-4o-transcribe",
    "openai-gpt-4o-mini-transcribe": "gpt-4o-mini-transcribe",
}


def build_adapters(model_keys: list[str], openai_api_key: str | None = None):
    """선택된 모델만 로드한다. 로컬 모델은 CPU 경합이 있으니(실측 확인됨)
    비교하고 싶은 모델만 골라서 켜야 한다."""
    adapters = []
    for key in model_keys:
        if key in OPENAI_API_MODEL_KEYS:
            adapters.append(OpenAIWhisperAdapter(model=OPENAI_API_MODEL_KEYS[key], api_key=openai_api_key))
        else:
            adapters.append(MODEL_REGISTRY[key]())
    return adapters


def detect_platform(url: str) -> str:
    if "chzzk.naver.com" in url:
        return "치지직"
    if "sooplive.co.kr" in url or "afreecatv.com" in url:
        return "SOOP"
    return "기타"


def fetch_stream_metadata(page_url: str) -> dict:
    """streamlink --json으로 방송 제목/스트리머명을 얻어본다 (플랫폼/버전에 따라
    비어있을 수 있음 — best-effort이며 실패해도 세션 시작 자체는 막지 않는다)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "streamlink", "--json", page_url],
            capture_output=True,
            text=True,
            timeout=20,
        )
        data = json.loads(result.stdout)
        metadata = data.get("metadata") or {}
        return {"streamer": metadata.get("author"), "title": metadata.get("title")}
    except Exception:
        return {"streamer": None, "title": None}


def _upload_segment_file(session_name: str, session_dir: Path, seq: int) -> None:
    """구간 하나(seq)의 완성된 영상/오디오 파일을 Supabase Storage에 올리고 URL을 기록한다.
    ffmpeg -f segment는 다음 구간 파일 쓰기를 시작해야 이전 파일이 완전히
    닫히므로, 이 함수는 항상 "그 다음 구간이 시작된 뒤"에만 호출해야 한다."""
    if CAPTURE_VIDEO:
        path = session_dir / f"video_{seq:05d}.mp4"
    else:
        path = session_dir / f"audio_{seq:05d}.m4a"
    if not path.exists():
        return
    url = db.upload_segment_file(session_name, path.name, path)
    if not url:
        return
    if CAPTURE_VIDEO:
        db.save_segment_url(session_name, seq, video_url=url)
    else:
        db.save_segment_url(session_name, seq, audio_url=url)


class StartRequest(BaseModel):
    url: str
    models: list[str]
    api_key: str | None = None


class VerifyKeyRequest(BaseModel):
    api_key: str


class GroundTruthRequest(BaseModel):
    session_name: str
    seq: int
    text: str


@app.get("/sessions")
async def get_sessions():
    records = db.load_sessions()
    for s in records:
        s["status"] = "분석중" if s["session_name"] in sessions else "분석완료"
    return {"sessions": records}


@app.get("/session/{session_name}")
async def get_session_detail(session_name: str):
    """세션 하나를 다시 볼 수 있도록 영상 구간 + 모델 결과 + 사람 정답을
    합쳐서 돌려준다 (라이브 중 프론트가 쓰는 것과 동일한 segments 구조).
    아직 진행 중인 세션이어도(그 세션을 시작한 사람이 새로고침한 경우 등)
    지금까지의 진행 상황을 그대로 복원할 수 있다.

    영상/오디오는 Supabase Storage 업로드가 끝난 구간은 그 URL을, 방금 만들어져
    아직 업로드 전인(진행 중 세션의) 최신 구간은 로컬 디스크 경로를 사용한다 —
    Render 재시작 시 로컬 디스크는 초기화되지만 그때는 세션도 이미 끝나 있으므로
    문제되지 않는다."""
    is_active = session_name in sessions
    segments: dict[int, dict] = {}

    if is_active:
        session_dir = SEGMENT_DIR / session_name
        if session_dir.exists():
            for video_file in sorted(session_dir.glob("video_*.mp4")):
                seq = int(video_file.stem.split("_")[1])
                segments[seq] = {"seq": seq, "video": f"/segments/{session_name}/{video_file.name}"}
            for audio_file in sorted(session_dir.glob("audio_*.m4a")):
                seq = int(audio_file.stem.split("_")[1])
                segments.setdefault(seq, {"seq": seq})
                segments[seq]["audio"] = f"/segments/{session_name}/{audio_file.name}"

    for seq, urls in db.load_segment_urls(session_name).items():
        segments.setdefault(seq, {"seq": seq})
        if "video" not in segments[seq] and urls.get("video_url"):
            segments[seq]["video"] = urls["video_url"]
        if "audio" not in segments[seq] and urls.get("audio_url"):
            segments[seq]["audio"] = urls["audio_url"]

    last_usage = None
    for entry in db.load_transcripts(session_name):
        seq = entry["seq"]
        segments.setdefault(seq, {"seq": seq})
        segments[seq]["text"] = entry["text"]
        segments[seq]["latency"] = entry["latency"]
        if entry.get("usage"):
            last_usage = entry["usage"]

    ground_truth = db.load_ground_truth(session_name)
    meta = next((s for s in db.load_sessions() if s["session_name"] == session_name), None)

    if meta is None and not segments:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")

    return {
        "meta": meta,
        "segments": sorted(segments.values(), key=lambda s: s["seq"]),
        "ground_truth": ground_truth,
        "usage": last_usage,
        "is_active": is_active,
    }


@app.get("/ground-truth/{session_name}")
async def get_ground_truth(session_name: str):
    return {"ground_truth": db.load_ground_truth(session_name)}


@app.post("/ground-truth")
async def save_ground_truth(req: GroundTruthRequest):
    """사람이 직접 들은 정답 텍스트를 구간(seq)별로 저장한다."""
    db.save_ground_truth(req.session_name, req.seq, req.text)
    return {"status": "saved"}


@app.post("/verify-openai-key")
async def verify_openai_key(req: VerifyKeyRequest):
    def check():
        OpenAI(api_key=req.api_key).models.list()

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, check)
        return {"valid": True}
    except AuthenticationError:
        return {"valid": False, "detail": "API 키가 올바르지 않습니다"}
    except Exception as e:
        return {"valid": False, "detail": str(e)}


async def run_capture_loop(capture: StreamCapture, orchestrator: Orchestrator, session_name: str):
    loop = asyncio.get_event_loop()
    session_dir = SEGMENT_DIR / session_name
    prev_seq: int | None = None

    async for seq, audio_chunk in capture.audio_chunks():
        if prev_seq is not None:
            # 직전 구간 파일은 ffmpeg가 이번 구간을 쓰기 시작한 시점에 이미 닫혀 있으므로
            # 이제 안전하게 Supabase Storage에 올릴 수 있다.
            loop.run_in_executor(None, _upload_segment_file, session_name, session_dir, prev_seq)
        prev_seq = seq

        session = sessions.get(session_name)
        if session is not None:
            session["last_seq"] = seq

        segment_payload = {"type": "segment", "seq": seq}
        if CAPTURE_VIDEO:
            segment_payload["video"] = f"/segments/{session_name}/video_{seq:05d}.mp4"
        else:
            segment_payload["audio"] = f"/segments/{session_name}/audio_{seq:05d}.m4a"
        await manager.broadcast(session_name, segment_payload)

        async def on_result(seq: int, model_name: str, text: str, latency: float, usage: dict | None = None):
            payload = {
                "type": "transcript",
                "seq": seq,
                "model": model_name,
                "text": text,
                "latency": round(latency, 2),
            }
            if usage:
                payload["usage"] = usage
            await manager.broadcast(session_name, payload)

            def persist():
                try:
                    db.save_transcript(session_name, seq, model_name, text, round(latency, 2), usage)
                except Exception:
                    logger.exception("transcript 저장 실패 (session=%s, seq=%s)", session_name, seq)

            loop.run_in_executor(None, persist)

        asyncio.create_task(orchestrator.process_chunk(seq, audio_chunk, on_result))


@app.post("/start")
async def start(req: StartRequest):
    known_keys = {*MODEL_REGISTRY, *OPENAI_API_MODEL_KEYS}
    unknown = [key for key in req.models if key not in known_keys]
    if unknown:
        raise HTTPException(status_code=400, detail=f"알 수 없는 모델: {unknown}")
    if not req.models:
        raise HTTPException(status_code=400, detail="최소 1개 모델을 선택해야 합니다")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 여러 명이 같은 초에 같은 모델로 동시에 시작할 수도 있으므로 임의 접미사로 겹치지 않게 한다.
    session_name = f"{timestamp}_{'-'.join(req.models)}_{uuid.uuid4().hex[:6]}"
    session_dir = SEGMENT_DIR / session_name

    loop = asyncio.get_event_loop()
    capture = StreamCapture(req.url, session_dir, capture_video=CAPTURE_VIDEO)
    await loop.run_in_executor(None, capture.start)
    metadata_future = loop.run_in_executor(None, fetch_stream_metadata, req.url)
    try:
        adapters = await loop.run_in_executor(None, build_adapters, req.models, req.api_key)
    except Exception as e:
        capture.stop()
        metadata_future.cancel()
        raise HTTPException(status_code=400, detail=str(e)) from e
    orchestrator = Orchestrator(adapters, SAMPLE_RATE)
    metadata = await metadata_future

    db.append_session(
        {
            "session_name": session_name,
            "url": req.url,
            "platform": detect_platform(req.url),
            "streamer": metadata["streamer"],
            "title": metadata["title"],
            "models": req.models,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
    )

    sessions[session_name] = {
        "capture": capture,
        "orchestrator": orchestrator,
        "last_seq": None,
        "task": asyncio.create_task(run_capture_loop(capture, orchestrator, session_name)),
    }

    return {"status": "started", "session_name": session_name}


@app.post("/stop/{session_name}")
async def stop(session_name: str):
    session = sessions.pop(session_name, None)
    if session is None:
        raise HTTPException(status_code=404, detail="이미 종료된 세션입니다")
    session["capture"].stop()
    session["task"].cancel()
    session["orchestrator"].shutdown()

    last_seq = session.get("last_seq")
    if last_seq is not None:
        # 마지막 구간은 "다음 구간이 시작될 때 이전 구간을 올린다"는 규칙으로는
        # 놓치므로, 세션이 끝나는 시점에 한 번 더 확실히 올려준다.
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _upload_segment_file, session_name, SEGMENT_DIR / session_name, last_seq)
    return {"status": "stopped"}


@app.websocket("/ws/{session_name}")
async def websocket_endpoint(ws: WebSocket, session_name: str):
    await manager.connect(session_name, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(session_name, ws)


SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/segments", StaticFiles(directory=str(SEGMENT_DIR)), name="segments")

# 배포 환경에서는 프론트 빌드 결과(frontend/dist)를 백엔드가 그대로 서빙한다
# (프론트 로컬 개발 서버는 이 디렉터리가 없으므로 영향 없음).
FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
