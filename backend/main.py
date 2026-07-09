import asyncio
import json
import logging
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

from adapters.moonshine_adapter import MoonshineAdapter
from adapters.openai_whisper_adapter import OpenAIWhisperAdapter
from adapters.sensevoice_adapter import SenseVoiceAdapter
from adapters.whisper_adapter import FasterWhisperAdapter
from capture import SAMPLE_RATE, StreamCapture
from orchestrator import Orchestrator

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt-compare")

LOG_DIR = Path(__file__).parent / "logs"
SEGMENT_DIR = Path(__file__).parent / "segments"

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

# 세션명 -> {"capture", "orchestrator", "log_file", "task"} — 동시에 여러 명이
# 각자 다른(또는 같은) 방송을 분석할 수 있도록 세션마다 독립적으로 상태를 갖는다.
# 인원 제한은 두지 않는다 (CPU 부담이 큰 로컬 모델을 여럿이 동시에 고르면
# 당연히 서로 느려지지만, 이건 사용자들이 알아서 모델을 고르며 관리할 몫이다).
sessions: dict[str, dict] = {}


MODEL_REGISTRY = {
    "large-v3": lambda: FasterWhisperAdapter("large-v3"),
    "large-v3-turbo": lambda: FasterWhisperAdapter("large-v3-turbo"),
    "medium": lambda: FasterWhisperAdapter("medium"),
    "sensevoice": lambda: SenseVoiceAdapter(),
    "moonshine": lambda: MoonshineAdapter(),
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


SESSIONS_PATH = LOG_DIR / "sessions.json"


def load_sessions() -> list:
    if SESSIONS_PATH.exists():
        with open(SESSIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def append_session(record: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    sessions_list = load_sessions()
    sessions_list.append(record)
    with open(SESSIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(sessions_list, f, ensure_ascii=False, indent=2)


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
    records = load_sessions()
    for s in records:
        s["status"] = "분석중" if s["session_name"] in sessions else "분석완료"
    return {"sessions": records}


@app.get("/session/{session_name}")
async def get_session_detail(session_name: str):
    """세션 하나를 다시 볼 수 있도록 영상 구간 + 모델 결과 + 사람 정답을
    합쳐서 돌려준다 (라이브 중 프론트가 쓰는 것과 동일한 segments 구조).
    아직 진행 중인 세션이어도(그 세션을 시작한 사람이 새로고침한 경우 등)
    지금까지의 진행 상황을 그대로 복원할 수 있다."""
    session_dir = SEGMENT_DIR / session_name
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")

    segments: dict[int, dict] = {}
    for video_file in sorted(session_dir.glob("video_*.mp4")):
        seq = int(video_file.stem.split("_")[1])
        segments[seq] = {"seq": seq, "video": f"/segments/{session_name}/{video_file.name}"}
    for audio_file in sorted(session_dir.glob("audio_*.m4a")):
        seq = int(audio_file.stem.split("_")[1])
        segments.setdefault(seq, {"seq": seq})
        segments[seq]["audio"] = f"/segments/{session_name}/{audio_file.name}"

    last_usage = None
    log_path = LOG_DIR / f"{session_name}.jsonl"
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                seq = entry["seq"]
                segments.setdefault(seq, {"seq": seq})
                segments[seq]["text"] = entry["text"]
                segments[seq]["latency"] = entry["latency"]
                if entry.get("usage"):
                    last_usage = entry["usage"]

    ground_truth = {}
    gt_path = LOG_DIR / f"{session_name}.ground_truth.json"
    if gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

    meta = next((s for s in load_sessions() if s["session_name"] == session_name), None)

    return {
        "meta": meta,
        "segments": sorted(segments.values(), key=lambda s: s["seq"]),
        "ground_truth": ground_truth,
        "usage": last_usage,
        "is_active": session_name in sessions,
    }


@app.get("/ground-truth/{session_name}")
async def get_ground_truth(session_name: str):
    path = LOG_DIR / f"{session_name}.ground_truth.json"
    if not path.exists():
        return {"ground_truth": {}}
    with open(path, "r", encoding="utf-8") as f:
        return {"ground_truth": json.load(f)}


@app.post("/ground-truth")
async def save_ground_truth(req: GroundTruthRequest):
    """사람이 직접 들은 정답 텍스트를 구간(seq)별로 저장한다.
    타이핑할 때마다 프론트에서 디바운스 호출하므로 매번 파일 전체를 덮어쓴다."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{req.session_name}.ground_truth.json"
    data = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    data[str(req.seq)] = req.text
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
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
    async for seq, audio_chunk in capture.audio_chunks():
        await manager.broadcast(
            session_name,
            {
                "type": "segment",
                "seq": seq,
                "video": f"/segments/{session_name}/video_{seq:05d}.mp4",
            },
        )

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
            session = sessions.get(session_name)
            log_file = session["log_file"] if session else None
            if log_file is not None:
                log_entry = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "seq": seq,
                    "model": model_name,
                    "text": text,
                    "latency": round(latency, 2),
                }
                if usage:
                    log_entry["usage"] = usage
                log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                log_file.flush()

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
    capture = StreamCapture(req.url, session_dir)
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

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{session_name}.jsonl"
    log_file = open(log_path, "a", encoding="utf-8")

    append_session(
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
        "log_file": log_file,
        "task": asyncio.create_task(run_capture_loop(capture, orchestrator, session_name)),
    }

    return {"status": "started", "log_file": str(log_path), "session_name": session_name}


@app.post("/stop/{session_name}")
async def stop(session_name: str):
    session = sessions.pop(session_name, None)
    if session is None:
        raise HTTPException(status_code=404, detail="이미 종료된 세션입니다")
    session["capture"].stop()
    session["task"].cancel()
    session["orchestrator"].shutdown()
    session["log_file"].close()
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
