"""Supabase(Postgres + Storage) 영구 저장소 래퍼.

Render 무료 플랜은 컨테이너 로컬 디스크가 재시작/재배포마다 초기화되므로,
세션 메타데이터·transcript·정답 텍스트·영상 구간 파일을 전부 Supabase에 둔다.

SUPABASE_URL/SUPABASE_KEY가 설정되지 않은 환경(예: 아직 Supabase를 세팅하지
않은 로컬 개발)에서는 조용히 아무 것도 하지 않는다 — 이 경우 결과는
지금까지처럼 세션이 끝나면 사라지지만, 최소한 앱 자체는 그대로 동작한다.
"""

import os

from supabase import Client, create_client

SEGMENT_BUCKET = "segments"

_client: Client | None = None
_client_loaded = False


def _get_client() -> Client | None:
    # main.py가 load_dotenv()를 호출하기 전에 이 모듈이 먼저 import될 수 있으므로,
    # 모듈 임포트 시점이 아니라 실제로 필요할 때 처음 한 번만 환경변수를 읽는다.
    global _client, _client_loaded
    if not _client_loaded:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if url and key:
            _client = create_client(url, key)
        _client_loaded = True
    return _client


def is_enabled() -> bool:
    return _get_client() is not None


def load_sessions() -> list[dict]:
    client = _get_client()
    if client is None:
        return []
    res = client.table("sessions").select("*").order("started_at", desc=True).execute()
    return res.data or []


def append_session(record: dict) -> None:
    client = _get_client()
    if client is None:
        return
    client.table("sessions").insert(record).execute()


def save_transcript(session_name: str, seq: int, model: str, text: str, latency: float, usage: dict | None) -> None:
    client = _get_client()
    if client is None:
        return
    client.table("transcript_segments").upsert(
        {
            "session_name": session_name,
            "seq": seq,
            "model": model,
            "text": text,
            "latency": latency,
            "usage": usage,
        },
        on_conflict="session_name,seq,model",
    ).execute()


def load_transcripts(session_name: str) -> list[dict]:
    client = _get_client()
    if client is None:
        return []
    res = (
        client.table("transcript_segments")
        .select("*")
        .eq("session_name", session_name)
        .order("created_at")
        .execute()
    )
    return res.data or []


def load_ground_truth(session_name: str) -> dict[str, str]:
    client = _get_client()
    if client is None:
        return {}
    res = client.table("ground_truths").select("*").eq("session_name", session_name).execute()
    return {str(row["seq"]): row["text"] for row in (res.data or [])}


def save_ground_truth(session_name: str, seq: int, text: str) -> None:
    client = _get_client()
    if client is None:
        return
    client.table("ground_truths").upsert(
        {"session_name": session_name, "seq": seq, "text": text},
        on_conflict="session_name,seq",
    ).execute()


def save_segment_url(session_name: str, seq: int, video_url: str | None = None, audio_url: str | None = None) -> None:
    """구간 파일의 Storage URL을 기록한다. video_url/audio_url 중 넘긴 값만
    갱신되고(PostgREST upsert가 페이로드에 있는 컬럼만 갈아끼움) 나머지는 그대로 남는다."""
    client = _get_client()
    if client is None:
        return
    record = {"session_name": session_name, "seq": seq}
    if video_url is not None:
        record["video_url"] = video_url
    if audio_url is not None:
        record["audio_url"] = audio_url
    client.table("segment_files").upsert(record, on_conflict="session_name,seq").execute()


def load_segment_urls(session_name: str) -> dict[int, dict]:
    client = _get_client()
    if client is None:
        return {}
    res = client.table("segment_files").select("*").eq("session_name", session_name).execute()
    return {row["seq"]: row for row in (res.data or [])}


def upload_segment_file(session_name: str, filename: str, local_path) -> str | None:
    """영상/오디오 구간 파일을 Storage에 올리고 public URL을 돌려준다.
    실패해도(네트워크 문제 등) 세션 진행 자체는 막지 않도록 예외를 삼킨다 —
    업로드가 안 되면 그 구간은 세션 재시작 후 리뷰에서 영상만 빠질 뿐이다."""
    client = _get_client()
    if client is None:
        return None
    path = f"{session_name}/{filename}"
    try:
        with open(local_path, "rb") as f:
            content_type = "video/mp4" if filename.endswith(".mp4") else "audio/mp4"
            client.storage.from_(SEGMENT_BUCKET).upload(
                path, f.read(), {"content-type": content_type, "upsert": "true"}
            )
        return client.storage.from_(SEGMENT_BUCKET).get_public_url(path)
    except Exception:
        return None
