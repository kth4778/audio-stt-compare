import { useEffect, useRef, useState } from 'react'
import './App.css'

// 로컬 개발(Vite 5173 + 백엔드 8000, 서로 다른 오리진)에서는 localhost:8000을 직접
// 가리키고, 배포 환경(백엔드가 프론트 빌드 결과도 같이 서빙하는 같은 오리진)에서는
// 상대 경로/현재 호스트를 쓴다.
const BACKEND_HTTP = import.meta.env.DEV ? 'http://localhost:8000' : ''
const BACKEND_WS = import.meta.env.DEV
  ? 'ws://localhost:8000/ws'
  : `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws`

// 진행 중 세션의 구간은 백엔드 로컬 경로(상대경로)로, 지난 세션은 Supabase
// Storage 절대 URL로 온다 — 이미 절대 URL이면 그대로, 아니면 BACKEND_HTTP를 붙인다.
const resolveMediaUrl = (path) => (path.startsWith('http') ? path : `${BACKEND_HTTP}${path}`)

const MODELS = [
  { key: 'large-v3', label: 'faster-whisper (large-v3)' },
  { key: 'large-v3-turbo', label: 'faster-whisper (large-v3-turbo)' },
  { key: 'medium', label: 'faster-whisper (medium)' },
  { key: 'sensevoice', label: 'SenseVoice' },
  { key: 'moonshine', label: 'Moonshine-tiny-ko' },
]

const API_MODELS = [
  { key: 'openai-whisper-1', label: 'OpenAI Whisper API (whisper-1) — 과금 발생' },
  { key: 'openai-gpt-4o-transcribe', label: 'OpenAI GPT-4o Transcribe — 과금 발생' },
  { key: 'openai-gpt-4o-mini-transcribe', label: 'OpenAI GPT-4o mini Transcribe — 과금 발생' },
]

const MODELS_BY_ANALYSIS_MODE = { local: MODELS, api: API_MODELS }

// idle: 분석 시작 대기 / connecting: /start 요청+모델 로딩 중 / connected: 로딩 완료 직후 잠깐 표시 / running: 실행 중
const STATUS_LABEL = {
  idle: '분석 시작',
  connecting: '연결중...',
  connected: '연결완료!',
  running: '중지',
}

const PLATFORM_CLASS = { 치지직: 'chzzk', SOOP: 'soop' }

function formatDateTime(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString('ko-KR', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function App() {
  const [url, setUrl] = useState('')
  const [status, setStatus] = useState('idle')
  const isActive = status !== 'idle'

  // 확정된(현재 실행 중인) 설정
  const [selectedModel, setSelectedModel] = useState(MODELS[0].key)

  // 시작 모달 안에서 임시로 고르는 값 (확정 누르기 전까지는 적용 안 됨)
  const [showModal, setShowModal] = useState(false)
  const [pendingAnalysisMode, setPendingAnalysisMode] = useState('local')
  const [pendingModel, setPendingModel] = useState(MODELS[0].key)
  const [pendingApiKey, setPendingApiKey] = useState('')
  const [apiKeyCheck, setApiKeyCheck] = useState({ status: 'idle', detail: null }) // idle | checking | valid | invalid

  const [logPath, setLogPath] = useState(null)
  const [sessionName, setSessionName] = useState(null)
  const [wsConnected, setWsConnected] = useState(false)
  const [segments, setSegments] = useState({}) // seq -> { seq, video, text, latency }
  const [selectedSeq, setSelectedSeq] = useState(null) // 상세 패널에 열려있는 구간
  const wsRef = useRef(null)

  // 사람이 직접 들은 정답 텍스트 (모델 결과와 별개로 비교용으로 입력)
  const [groundTruths, setGroundTruths] = useState({}) // seq -> text
  const [groundTruthStatus, setGroundTruthStatus] = useState({}) // seq -> 'saving' | 'saved'
  const groundTruthTimers = useRef({})

  const [pastSessions, setPastSessions] = useState([])

  // OpenAI API 분석일 때만 채워지는 실시간 사용량(요청 수/오디오 길이/추정 비용)
  const [apiUsage, setApiUsage] = useState(null)

  const loadPastSessions = () => {
    fetch(`${BACKEND_HTTP}/sessions`)
      .then((res) => res.json())
      .then((data) => setPastSessions(data.sessions ?? []))
      .catch(() => {})
  }

  const loadGroundTruths = (name) => {
    fetch(`${BACKEND_HTTP}/ground-truth/${encodeURIComponent(name)}`)
      .then((res) => res.json())
      .then((data) => setGroundTruths(data.ground_truth ?? {}))
      .catch(() => {})
  }

  // 세션별 WebSocket — 여러 명이 각자 다른 세션을 동시에 볼 수 있으므로 항상
  // 특정 세션 이름에 붙는다. wsSessionRef로 "지금 붙어야 할 세션"을 추적해서,
  // 세션을 나가거나 바꾼 뒤에도 이전 연결이 뒤늦게 재연결을 시도하지 않게 한다.
  const wsSessionRef = useRef(null)

  const connectSessionWs = (name) => {
    wsSessionRef.current = name
    const open = () => {
      if (wsSessionRef.current !== name) return
      const ws = new WebSocket(`${BACKEND_WS}/${encodeURIComponent(name)}`)
      wsRef.current = ws

      ws.onopen = () => setWsConnected(true)
      ws.onclose = () => {
        setWsConnected(false)
        if (wsSessionRef.current === name) {
          // 백엔드 재시작 등으로 끊겨도 사람이 새로고침 안 해도 되도록 자동 재연결.
          setTimeout(open, 2000)
        }
      }
      ws.onerror = () => ws.close()
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data)
        if (msg.type === 'segment') {
          setSegments((prev) => ({
            ...prev,
            [msg.seq]: { ...prev[msg.seq], seq: msg.seq, video: msg.video, audio: msg.audio },
          }))
        } else if (msg.type === 'transcript') {
          setSegments((prev) => ({
            ...prev,
            [msg.seq]: { ...prev[msg.seq], seq: msg.seq, text: msg.text, latency: msg.latency },
          }))
          if (msg.usage) setApiUsage(msg.usage)
        }
      }
    }
    open()
  }

  const disconnectSessionWs = () => {
    wsSessionRef.current = null
    wsRef.current?.close()
    wsRef.current = null
    setWsConnected(false)
  }

  const openPastSession = async (name, { openFirst = true, resumeOnly = false } = {}) => {
    try {
      const res = await fetch(`${BACKEND_HTTP}/session/${encodeURIComponent(name)}`)
      if (!res.ok) {
        if (resumeOnly) localStorage.removeItem('activeSessionName')
        return
      }
      const data = await res.json()
      if (resumeOnly && !data.is_active) {
        // 새로고침 시점엔 이미 남이 끝냈거나 내가 끝낸 세션 — 조용히 정리만 하고 홈 화면 유지.
        localStorage.removeItem('activeSessionName')
        return
      }
      const segMap = {}
      for (const seg of data.segments) segMap[seg.seq] = seg
      setSegments(segMap)
      setGroundTruths(data.ground_truth ?? {})
      setGroundTruthStatus({})
      setSessionName(name)
      setSelectedModel(data.meta?.models?.[0] ?? selectedModel)
      if (openFirst) setSelectedSeq(data.segments[0]?.seq ?? null)
      setLogPath(null)
      setApiUsage(data.usage ?? null)
      if (data.is_active) {
        localStorage.setItem('activeSessionName', name)
        setStatus('running')
        connectSessionWs(name)
      } else {
        localStorage.removeItem('activeSessionName')
        setStatus('idle')
      }
    } catch {
      // 무시 — 카드 클릭이 실패해도 홈 화면은 그대로 유지된다
    }
  }

  const handleGoHome = () => {
    disconnectSessionWs()
    localStorage.removeItem('activeSessionName')
    setSegments({})
    setSelectedSeq(null)
    setGroundTruths({})
    setGroundTruthStatus({})
    setSessionName(null)
    setLogPath(null)
    setApiUsage(null)
    loadPastSessions()
  }

  useEffect(() => {
    loadPastSessions()
    // 새로고침해도 "내가" 돌리고 있던 세션이 아직 살아있으면 다시 붙어서
    // 중지 버튼과 그동안의 결과를 복원한다 (localStorage에 세션명을 남겨뒀다가 확인).
    const savedName = localStorage.getItem('activeSessionName')
    if (savedName) openPastSession(savedName, { openFirst: false, resumeOnly: true })
    return () => disconnectSessionWs()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const orderedSeqs = Object.keys(segments)
    .map(Number)
    .sort((a, b) => a - b)

  const pendingCount = orderedSeqs.filter((seq) => segments[seq].text == null).length

  // 새 구간이 도착해도 목록을 강제로 스크롤하지 않는다 — 사용자가 과거 구간을 보고 있을 때
  // 화면이 임의로 하단으로 넘어가면 방해가 되므로 (사용자 피드백 반영).

  const selectedIndex = orderedSeqs.indexOf(selectedSeq)
  const selected = selectedSeq !== null ? segments[selectedSeq] : null

  const goPrev = () => {
    if (selectedIndex > 0) setSelectedSeq(orderedSeqs[selectedIndex - 1])
  }
  const goNext = () => {
    if (selectedIndex >= 0 && selectedIndex < orderedSeqs.length - 1) {
      setSelectedSeq(orderedSeqs[selectedIndex + 1])
    }
  }

  const openStartModal = () => {
    if (!url) return
    const isApiModel = API_MODELS.some((m) => m.key === selectedModel)
    setPendingAnalysisMode(isApiModel ? 'api' : 'local')
    setPendingModel(selectedModel)
    setPendingApiKey('')
    setApiKeyCheck({ status: 'idle', detail: null })
    setShowModal(true)
  }

  const handleVerifyApiKey = async () => {
    if (!pendingApiKey) return
    setApiKeyCheck({ status: 'checking', detail: null })
    try {
      const res = await fetch(`${BACKEND_HTTP}/verify-openai-key`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: pendingApiKey }),
      })
      const data = await res.json()
      setApiKeyCheck({ status: data.valid ? 'valid' : 'invalid', detail: data.detail ?? null })
    } catch (err) {
      setApiKeyCheck({ status: 'invalid', detail: `백엔드에 연결할 수 없습니다: ${err.message}` })
    }
  }

  const canConfirmStart = pendingAnalysisMode !== 'api' || apiKeyCheck.status === 'valid'

  const handleConfirmStart = async () => {
    if (!canConfirmStart) return
    setShowModal(false)
    setSelectedModel(pendingModel)
    setSegments({})
    setSelectedSeq(null)
    setGroundTruths({})
    setGroundTruthStatus({})
    setApiUsage(null)
    setStatus('connecting')
    try {
      const res = await fetch(`${BACKEND_HTTP}/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url,
          models: [pendingModel],
          api_key: pendingAnalysisMode === 'api' ? pendingApiKey : null,
        }),
      })
      const data = await res.json()
      if (!res.ok) {
        alert(`시작 실패: ${data.detail ?? res.status}`)
        setStatus('idle')
        return
      }
      setLogPath(data.log_file ?? null)
      setSessionName(data.session_name ?? null)
      if (data.session_name) {
        loadGroundTruths(data.session_name)
        localStorage.setItem('activeSessionName', data.session_name)
        connectSessionWs(data.session_name)
      }
      setStatus('connected')
      setTimeout(() => setStatus('running'), 1200)
      loadPastSessions()
    } catch (err) {
      alert(`백엔드에 연결할 수 없습니다: ${err.message}`)
      setStatus('idle')
    }
  }

  const handleStop = async () => {
    if (sessionName) {
      await fetch(`${BACKEND_HTTP}/stop/${encodeURIComponent(sessionName)}`, { method: 'POST' })
    }
    disconnectSessionWs()
    localStorage.removeItem('activeSessionName')
    setStatus('idle')
  }

  const saveGroundTruth = async (seq, text) => {
    if (!sessionName) return
    setGroundTruthStatus((prev) => ({ ...prev, [seq]: 'saving' }))
    try {
      await fetch(`${BACKEND_HTTP}/ground-truth`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_name: sessionName, seq, text }),
      })
      setGroundTruthStatus((prev) => ({ ...prev, [seq]: 'saved' }))
    } catch {
      setGroundTruthStatus((prev) => ({ ...prev, [seq]: null }))
    }
  }

  // 타이핑 중엔 저장 버튼 없이 입력을 멈추면(800ms) 자동 저장한다.
  const handleGroundTruthChange = (seq, text) => {
    setGroundTruths((prev) => ({ ...prev, [seq]: text }))
    clearTimeout(groundTruthTimers.current[seq])
    groundTruthTimers.current[seq] = setTimeout(() => saveGroundTruth(seq, text), 800)
  }

  // 입력창을 클릭(포커스)하는 것 자체를 "검수 완료"로 본다 — 스트리머가 말을
  // 안 한 구간은 애초에 고칠 게 없으니 클릭만으로 끝나야 하기 때문. 아직 저장된
  // 적 없는 구간에 한해, 기본으로 채워진 모델 텍스트를 그대로 정답으로 저장한다.
  const handleGroundTruthFocus = (seq) => {
    if (Object.prototype.hasOwnProperty.call(groundTruths, seq)) return
    const modelText = segments[seq]?.text
    if (modelText == null) return
    setGroundTruths((prev) => ({ ...prev, [seq]: modelText }))
    saveGroundTruth(seq, modelText)
  }

  const flushGroundTruth = (seq) => {
    clearTimeout(groundTruthTimers.current[seq])
    saveGroundTruth(seq, groundTruths[seq] ?? '')
  }

  return (
    <div className="page">
      <div className="url-bar">
        <span className={`ws-indicator ${wsConnected ? 'connected' : 'disconnected'}`}>
          {wsConnected ? '● 연결됨' : '● 연결 끊김'}
        </span>
        <input
          type="text"
          placeholder="치지직 / SOOP 방송 URL 입력"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={isActive}
        />
        {status === 'running' ? (
          <button onClick={handleStop}>{STATUS_LABEL.running}</button>
        ) : (
          <button onClick={openStartModal} disabled={status !== 'idle' || !url} className={`status-${status}`}>
            {STATUS_LABEL[status]}
          </button>
        )}
        <a
          href={url || undefined}
          target="_blank"
          rel="noreferrer"
          className={`view-original-btn${url ? '' : ' disabled'}`}
        >
          원본 방송 보기
        </a>
        {!isActive && orderedSeqs.length > 0 && (
          <button className="home-btn" onClick={handleGoHome}>
            홈으로
          </button>
        )}
      </div>

      {logPath && <div className="log-path">기록 파일: {logPath}</div>}

      {orderedSeqs.length === 0 && !isActive ? (
        <div className="home-page">
          <div className="home-page-header">
            <div className="home-page-title">분석한 방송</div>
            <span className="home-page-count">{pastSessions.length}개</span>
          </div>
          {pastSessions.length === 0 ? (
            <div className="home-empty">아직 분석한 방송이 없습니다. 위에서 URL을 입력하고 분석을 시작하세요.</div>
          ) : (
            <div className="session-grid">
              {[...pastSessions].reverse().map((s) => (
                <div className="session-card" key={s.session_name} onClick={() => openPastSession(s.session_name)}>
                  <div className="session-card-badges">
                    <span className={`session-card-platform ${PLATFORM_CLASS[s.platform] ?? ''}`}>{s.platform}</span>
                    <span className={`session-card-status ${s.status === '분석중' ? 'active' : 'done'}`}>
                      {s.status}
                    </span>
                  </div>
                  <div className="session-card-streamer">{s.streamer ?? '스트리머 이름 알 수 없음'}</div>
                  <div className="session-card-title">{s.title ?? '(방송 제목 없음)'}</div>
                  <div className="session-card-models">
                    {s.models.map((m) => (
                      <span className="session-card-model-pill" key={m}>
                        {m}
                      </span>
                    ))}
                  </div>
                  <div className="session-card-footer">
                    <span className="session-card-time">{formatDateTime(s.started_at)}</span>
                    <a
                      href={s.url}
                      target="_blank"
                      rel="noreferrer"
                      className="session-card-original-link"
                      onClick={(e) => e.stopPropagation()}
                    >
                      원본 보기
                    </a>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="chat-page">
          <div className="chat-pane">
            <div className="chat-pane-header">
              결과 목록{isActive && <span className="live-dot" title="실시간 수신 중" />}
            </div>
            <div className="pending-card">
              <span className="pending-dot" />
              처리 대기 중인 구간 {pendingCount}개
            </div>
            {apiUsage && (
              <div className="api-usage-card">
                OpenAI API · 요청 {apiUsage.request_count}건 · 오디오{' '}
                {(apiUsage.total_audio_seconds / 60).toFixed(1)}분 · 추정 비용 $
                {apiUsage.estimated_cost_usd.toFixed(4)}
              </div>
            )}
            <div className="chat-list">
              {[...orderedSeqs]
                .reverse()
                .filter((seq) => segments[seq].text != null)
                .map((seq) => {
                  const seg = segments[seq]
                  // 빈 문자열로 저장한 것도 "말을 안 함"으로 검수 완료한 것이므로,
                  // 텍스트 내용이 아니라 저장된 적이 있는지(키 존재 여부)로 판단한다.
                  const hasGroundTruth = Object.prototype.hasOwnProperty.call(groundTruths, seq)
                  return (
                    <div
                      className={`chat-row${seq === selectedSeq ? ' active' : ''}`}
                      key={seq}
                      onClick={() => setSelectedSeq(seq)}
                    >
                      <div className="chat-avatar-wrap">
                        <div className="chat-avatar">{selectedModel.slice(0, 1).toUpperCase()}</div>
                        <span
                          className={`chat-status-dot ${hasGroundTruth ? 'green' : 'red'}`}
                          title={hasGroundTruth ? '사람이 검수함' : '아직 검수 안 함'}
                        />
                      </div>
                      <div className="chat-content">
                        <div className="chat-meta">
                          <span className="chat-seq">#{seq}</span>
                          {seg.latency != null && <span className="chat-latency">{seg.latency}s</span>}
                        </div>
                        <div className="chat-bubble">{seg.text}</div>
                      </div>
                    </div>
                  )
                })}
            </div>
          </div>

          {selected && (
            <div className="detail-pane">
              <div className="detail-pane-header">
                <button onClick={goPrev} disabled={selectedIndex <= 0}>
                  ← 이전
                </button>
                <span>#{selected.seq}</span>
                <button onClick={goNext} disabled={selectedIndex < 0 || selectedIndex >= orderedSeqs.length - 1}>
                  다음 →
                </button>
                <button className="detail-close-btn" onClick={() => setSelectedSeq(null)}>
                  닫기
                </button>
              </div>
              <div className="detail-pane-body">
                {selected.video && (
                  <video
                    key={selected.video}
                    src={resolveMediaUrl(selected.video)}
                    controls
                    autoPlay
                    loop
                    className="segment-video"
                  />
                )}
                {!selected.video && selected.audio && (
                  <audio
                    key={selected.audio}
                    src={resolveMediaUrl(selected.audio)}
                    controls
                    autoPlay
                    loop
                    className="segment-audio"
                  />
                )}
                <div className="segment-text-card">
                  <span className="model-pill">{selectedModel}</span>
                  {selected.latency != null && <span className="segment-latency-stat">{selected.latency}s</span>}
                  <p>{selected.text ?? '(모델 처리 중...)'}</p>
                </div>
                <div className="ground-truth-card">
                  <div className="ground-truth-header">
                    <span>정답 텍스트 (실제로 들은 내용)</span>
                    {groundTruthStatus[selected.seq] === 'saving' && (
                      <span className="ground-truth-state saving">저장 중...</span>
                    )}
                    {groundTruthStatus[selected.seq] === 'saved' && (
                      <span className="ground-truth-state saved">저장됨</span>
                    )}
                  </div>
                  <textarea
                    className="ground-truth-input"
                    placeholder="이 구간에서 실제로 들린 말을 입력하세요 (입력을 멈추면 자동 저장됩니다)"
                    value={groundTruths[selected.seq] ?? selected.text ?? ''}
                    onFocus={() => handleGroundTruthFocus(selected.seq)}
                    onChange={(e) => handleGroundTruthChange(selected.seq, e.target.value)}
                    onBlur={() => flushGroundTruth(selected.seq)}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {showModal && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <h3>분석 설정</h3>

            <div className="modal-section">
              <div className="modal-section-title">분석 방식</div>
              <label className="modal-option">
                <input
                  type="radio"
                  name="pendingAnalysisMode"
                  checked={pendingAnalysisMode === 'local'}
                  onChange={() => {
                    setPendingAnalysisMode('local')
                    setPendingModel(MODELS[0].key)
                    setApiKeyCheck({ status: 'idle', detail: null })
                  }}
                />
                로컬(현재 서버)에서 실행
              </label>
              <label className="modal-option">
                <input
                  type="radio"
                  name="pendingAnalysisMode"
                  checked={pendingAnalysisMode === 'api'}
                  onChange={() => {
                    setPendingAnalysisMode('api')
                    setPendingModel(API_MODELS[0].key)
                  }}
                />
                API로 분석 (OpenAI Whisper API)
              </label>

              {pendingAnalysisMode === 'api' && (
                <div className="api-key-box">
                  <input
                    type="password"
                    placeholder="OpenAI API 키 붙여넣기 (sk-...)"
                    value={pendingApiKey}
                    onChange={(e) => {
                      setPendingApiKey(e.target.value)
                      setApiKeyCheck({ status: 'idle', detail: null })
                    }}
                    className="api-key-input"
                  />
                  <button
                    type="button"
                    className="api-key-verify-btn"
                    onClick={handleVerifyApiKey}
                    disabled={!pendingApiKey || apiKeyCheck.status === 'checking'}
                  >
                    {apiKeyCheck.status === 'checking' ? '확인중...' : '연결'}
                  </button>
                  {apiKeyCheck.status === 'valid' && (
                    <div className="api-key-status valid">키 확인 완료</div>
                  )}
                  {apiKeyCheck.status === 'invalid' && (
                    <div className="api-key-status invalid">{apiKeyCheck.detail ?? '키가 올바르지 않습니다'}</div>
                  )}
                </div>
              )}
            </div>

            <div className="modal-section">
              <div className="modal-section-title">모델 선택</div>
              {MODELS_BY_ANALYSIS_MODE[pendingAnalysisMode].map((model) => (
                <label key={model.key} className="modal-option">
                  <input
                    type="radio"
                    name="pendingModel"
                    checked={pendingModel === model.key}
                    onChange={() => setPendingModel(model.key)}
                  />
                  {model.label}
                </label>
              ))}
            </div>

            <div className="modal-actions">
              <button className="modal-cancel-btn" onClick={() => setShowModal(false)}>
                취소
              </button>
              <button className="modal-confirm-btn" onClick={handleConfirmStart} disabled={!canConfirmStart}>
                확정
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
