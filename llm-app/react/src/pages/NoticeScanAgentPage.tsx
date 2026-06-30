import { useEffect, useMemo, useRef, useState } from "react"
import { useAppState } from "../context/appStateStore"
import { uploadFile } from "../api/upload"
import { analyzeNoticeStreaming } from "../api/qna"
import { Spinner } from "../components/common/Spinner"
import { PipelineStrip, type PipelineStep, type StepStatus } from "../components/notice/PipelineStrip"
import { DonutSummary, ExtractionDashboard } from "../components/notice/ExtractionDashboard"
import type { AgentProgressEvent, NoticeScanResult } from "../types"

type PageStatus = "idle" | "uploading" | "analyzing" | "done" | "error"

interface ProgressStep {
  step: string
  label: string
  status: "running" | "done" | "failed"
  message: string
}

// ─── Pipeline step computation ────────────────────────────────────────────────

function computePipeline(pageStatus: PageStatus, progressSteps: ProgressStep[]): PipelineStep[] {
  const getStep = (id: string) => progressSteps.find((s) => s.step === id)

  const toStatus = (step?: ProgressStep, fallback: StepStatus = "waiting"): StepStatus => {
    if (!step) return fallback
    if (step.status === "running") return "running"
    if (step.status === "done") return "done"
    if (step.status === "failed") return "failed"
    return "waiting"
  }

  const uploadStatus: StepStatus =
    pageStatus === "idle" || pageStatus === "error" ? "waiting" :
    pageStatus === "uploading" ? "running" : "done"

  const convertStatus: StepStatus =
    pageStatus === "analyzing" || pageStatus === "done"
      ? toStatus(getStep("convert_to_markdown"), pageStatus === "done" ? "done" : "waiting")
      : "waiting"

  const extractStatus: StepStatus =
    pageStatus === "analyzing" || pageStatus === "done"
      ? toStatus(getStep("extract_metadata"), "waiting")
      : "waiting"

  const doneStatus: StepStatus = pageStatus === "done" ? "done" : "waiting"

  return [
    { label: "업로드",   sub: "공고서 파일",       status: uploadStatus },
    { label: "문서 변환", sub: "레이아웃 · OCR",   status: convertStatus },
    { label: "정보 추출", sub: "Structured Output", status: extractStatus },
    { label: "완료",      sub: "분석 결과 확인",    status: doneStatus },
  ]
}

// ─── Idle / Upload UI ─────────────────────────────────────────────────────────

function UploadZone({
  pageStatus,
  fileName,
  errorMsg,
  onFileChange,
}: {
  pageStatus: PageStatus
  fileName: string | null
  errorMsg: string
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void
}) {
  const isLoading = pageStatus === "uploading" || pageStatus === "analyzing"
  const fileRef = useRef<HTMLInputElement>(null)

  return (
    <div style={{ display: "flex", flex: 1, alignItems: "center", justifyContent: "center", background: "#F4F6F9" }}>
      <div style={{ width: "100%", maxWidth: 520, background: "#fff", borderRadius: 20, border: "1px solid #E6E9EF", boxShadow: "0 2px 16px rgba(20,26,34,.07)", padding: "40px 36px" }}>

        {/* Brand header */}
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 28 }}>
          <div style={{ width: 36, height: 36, borderRadius: 10, background: "#161A22", display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontWeight: 700, fontSize: 16, letterSpacing: "-.5px" }}>V</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: "-.2px", color: "#161A22" }}>비축공고서 추출 엔진</div>
            <div style={{ fontSize: 11.5, color: "#8A93A6" }}>Structured Output Agent · v2.4</div>
          </div>
        </div>

        {/* Drop zone */}
        <label
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 12,
            borderRadius: 14,
            border: `2px dashed ${isLoading ? "#3457D5" : "#D0D5DF"}`,
            background: isLoading ? "#F4F7FE" : "#FAFBFC",
            padding: "40px 24px",
            cursor: isLoading ? "default" : "pointer",
            transition: "border-color .2s, background .2s",
          }}
        >
          <input ref={fileRef} type="file" accept=".pdf,.html,.htm" style={{ display: "none" }} onChange={onFileChange} disabled={isLoading} />
          {isLoading ? (
            <Spinner className="size-8 text-[#3457D5]" />
          ) : (
            <div style={{ width: 44, height: 44, borderRadius: 12, background: "#EAF0FE", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#3457D5" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
          )}
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: "#1A1F29", marginBottom: 4 }}>
              {isLoading
                ? pageStatus === "uploading" ? "파일을 저장하는 중..." : "공고서를 분석하는 중..."
                : "공고문 파일을 업로드하세요"}
            </div>
            <div style={{ fontSize: 12, color: "#9AA3B2" }}>
              {isLoading ? (fileName ?? "") : "PDF, HTML 파일 · 클릭하여 선택"}
            </div>
          </div>
        </label>

        {/* Status / error message */}
        {pageStatus === "error" && (
          <div style={{ marginTop: 16, padding: "10px 14px", borderRadius: 10, background: "#FEF2F2", border: "1px solid #FECACA", fontSize: 13, color: "#DC2626" }}>
            ⚠️ {errorMsg}
          </div>
        )}

        {pageStatus === "idle" && (
          <p style={{ marginTop: 16, fontSize: 12, color: "#B0B8C8", textAlign: "center" }}>
            파일을 선택하면 자동으로 분석을 시작합니다.
          </p>
        )}
      </div>
    </div>
  )
}

// ─── Analyzing overlay (pipeline strip + progress) ────────────────────────────

function AnalyzingView({ progressSteps, pipeline }: { progressSteps: ProgressStep[]; pipeline: PipelineStep[] }) {
  return (
    <div style={{ display: "flex", flex: 1, alignItems: "center", justifyContent: "center", background: "#F4F6F9", padding: 24 }}>
      <div style={{ width: "100%", maxWidth: 480, display: "flex", flexDirection: "column", gap: 16 }}>

        {/* Live pipeline mini view */}
        <div style={{ background: "#fff", borderRadius: 14, border: "1px solid #E6E9EF", padding: "18px 20px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 18 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#0E9F6E", animation: "dotPulse 1.6s ease-out infinite", display: "inline-block" }} />
            <span style={{ fontSize: 13, fontWeight: 700, color: "#161A22" }}>추출 엔진 가동 중</span>
          </div>

          {progressSteps.length === 0 ? (
            <div style={{ display: "flex", alignItems: "center", gap: 10, color: "#8A93A6", fontSize: 13 }}>
              <Spinner className="size-4 text-[#3457D5]" />
              <span>파이프라인을 초기화하는 중...</span>
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {progressSteps.map((step) => (
                <div key={step.step} style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
                  {step.status === "running" ? (
                    <div style={{ marginTop: 1, width: 18, height: 18, border: "2px solid #BFD0F6", borderTopColor: "#3457D5", borderRadius: "50%", animation: "spin .7s linear infinite", flexShrink: 0 }} />
                  ) : step.status === "done" ? (
                    <div style={{ marginTop: 1, width: 18, height: 18, borderRadius: "50%", background: "#E3F6EE", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#0E9F6E" strokeWidth="3"><path d="M20 6 9 17l-5-5" /></svg>
                    </div>
                  ) : (
                    <div style={{ marginTop: 1, width: 18, height: 18, borderRadius: "50%", background: "#FCF2E3", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#E07B12" strokeWidth="3"><path d="M18 6 6 18M6 6l12 12" /></svg>
                    </div>
                  )}
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: step.status === "running" ? "#3457D5" : "#161A22" }}>{step.label}</div>
                    {step.message && <div style={{ fontSize: 11.5, color: "#9AA3B2", marginTop: 1 }}>{step.message}</div>}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Pipeline steps summary */}
        <div style={{ background: "#fff", borderRadius: 14, border: "1px solid #E6E9EF", padding: "14px 18px" }}>
          <div style={{ fontSize: 11, color: "#8A93A6", marginBottom: 10 }}>처리 단계</div>
          <div style={{ display: "flex", alignItems: "center" }}>
            {pipeline.map((step, i) => (
              <div key={step.label} style={{ display: "flex", alignItems: "center", flex: i < pipeline.length - 1 ? "auto" : "none" }}>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4, width: 68 }}>
                  <div style={{
                    width: 24, height: 24, borderRadius: "50%",
                    background: step.status === "done" ? "#E3F6EE" : step.status === "running" ? "#EAF0FE" : "#F1F3F7",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    ...(step.status === "running" ? { animation: "ringPulse 1.8s ease-out infinite" } : {}),
                  }}>
                    {step.status === "done" ? (
                      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#0E9F6E" strokeWidth="3"><path d="M20 6 9 17l-5-5" /></svg>
                    ) : step.status === "running" ? (
                      <div style={{ width: 11, height: 11, border: "1.8px solid #BFD0F6", borderTopColor: "#3457D5", borderRadius: "50%", animation: "spin .7s linear infinite" }} />
                    ) : (
                      <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#CDD2DB" }} />
                    )}
                  </div>
                  <span style={{ fontSize: 10, fontWeight: 600, color: step.status === "running" ? "#3457D5" : "#8A93A6", textAlign: "center", whiteSpace: "nowrap" }}>{step.label}</span>
                </div>
                {i < pipeline.length - 1 && (
                  <div style={{ flex: 1, height: 2, minWidth: 8, marginBottom: 16, background: step.status === "done" ? "#D7F0E4" : "#E6E9EF" }} />
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export function NoticeScanAgentPage() {
  const { selectedAgent, convrstnId, messages, refreshHistory } = useAppState()
  const [pageStatus, setPageStatus] = useState<PageStatus>("idle")
  const [fileName, setFileName] = useState<string | null>(null)
  const [result, setResult] = useState<NoticeScanResult | null>(null)
  const [progressSteps, setProgressSteps] = useState<ProgressStep[]>([])
  const [errorMsg, setErrorMsg] = useState("")
  const [elapsedSec, setElapsedSec] = useState<number | null>(null)
  const startTimeRef = useRef<number>(0)

  // Restore from conversation history
  useEffect(() => {
    if (messages.length === 0) {
      setPageStatus("idle"); setFileName(null); setResult(null); setProgressSteps([]); setElapsedSec(null)
      return
    }
    const last = messages[messages.length - 1]
    if (!last.answer) return
    try {
      const parsed = JSON.parse(last.answer) as NoticeScanResult
      setFileName(last.question.split(/[/\\]/).pop() ?? last.question)
      setResult(parsed); setPageStatus("done"); setProgressSteps([]); setElapsedSec(null)
    } catch { /* not a notice result */ }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [convrstnId])

  const pipelineSteps = useMemo(() => computePipeline(pageStatus, progressSteps), [pageStatus, progressSteps])

  if (!selectedAgent) return null

  const reset = () => {
    setPageStatus("idle"); setFileName(null); setResult(null); setProgressSteps([]); setErrorMsg(""); setElapsedSec(null)
  }

  const handleProgressEvent = (event: AgentProgressEvent) => {
    if (event.type !== "progress" || !event.step) return
    setProgressSteps((prev) => {
      const idx = prev.findIndex((s) => s.step === event.step)
      const next: ProgressStep = { step: event.step!, label: event.label ?? event.step!, status: event.status ?? "running", message: event.message ?? "" }
      if (idx >= 0) { const a = [...prev]; a[idx] = next; return a }
      return [...prev, next]
    })
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ""
    if (!file) return

    setFileName(file.name); setResult(null); setProgressSteps([]); setErrorMsg("")
    setPageStatus("uploading")

    try {
      const uploaded = await uploadFile(file)
      setPageStatus("analyzing")
      startTimeRef.current = Date.now()

      const res = await analyzeNoticeStreaming(
        { agent_id: selectedAgent.agent_id, agent_mode: selectedAgent.mode, convrstnId, fileFullPath: uploaded.fileFullPath, question: "", enableExtDocse: false },
        handleProgressEvent,
      )

      const elapsed = Math.round((Date.now() - startTimeRef.current) / 100) / 10
      setElapsedSec(elapsed)
      setResult(res); setPageStatus("done")
      refreshHistory()
    } catch {
      setPageStatus("error"); setErrorMsg("분석 중 오류가 발생했습니다. 다시 시도해 주세요.")
    }
  }

  const isDashboard = pageStatus === "done" && result?.extracted_data

  // Count for donut summary
  const okCount = useMemo(() => {
    if (!result?.extracted_data) return 0
    const g = result.extracted_data.general ?? {}
    const e = result.extracted_data.execution ?? {}
    const all = { ...g, ...e }
    return Object.values(all).filter((v) => v !== null && v !== undefined && v !== "").length
  }, [result])
  const totalFields = 21
  const reviewCount = totalFields - okCount

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden", background: "#F4F6F9", fontFamily: "Pretendard, system-ui, sans-serif", color: "#161A22", WebkitFontSmoothing: "antialiased" }}>

      {/* ── Top header bar ── */}
      <header style={{ flexShrink: 0, background: "#FFFFFF", borderBottom: "1px solid #E6E9EF", display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 22px", height: 56 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{ width: 28, height: 28, borderRadius: 8, background: "#161A22", display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontWeight: 700, fontSize: 13 }}>V</div>
          <div style={{ lineHeight: 1.25 }}>
            <div style={{ fontSize: 13.5, fontWeight: 700, letterSpacing: "-.2px" }}>비축공고서 추출 엔진</div>
            <div style={{ fontSize: 11, color: "#8A93A6" }}>Structured Output Agent · v2.4</div>
          </div>
          {fileName && (
            <div style={{ display: "flex", alignItems: "center", gap: 7, marginLeft: 8, padding: "5px 11px", background: "#F4F6F9", border: "1px solid #E6E9EF", borderRadius: 7 }}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#5B6577" strokeWidth="2"><path d="M14 3v4a1 1 0 0 0 1 1h4" /><path d="M5 21V5a2 2 0 0 1 2-2h8l5 5v13a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2z" /></svg>
              <span style={{ fontSize: 11.5, fontFamily: "'IBM Plex Mono', monospace", color: "#5B6577" }}>{fileName}</span>
            </div>
          )}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          {pageStatus === "analyzing" && (
            <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "5px 13px", background: "#E3F6EE", border: "1px solid #BCEAD4", borderRadius: 20 }}>
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#0E9F6E", animation: "dotPulse 1.6s ease-out infinite", display: "inline-block" }} />
              <span style={{ fontSize: 12.5, fontWeight: 600, color: "#0B7C56" }}>추출 엔진 가동 중</span>
            </div>
          )}
          {pageStatus === "done" && (
            <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "5px 13px", background: "#EAF0FE", border: "1px solid #C8D6F7", borderRadius: 20 }}>
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#3457D5" strokeWidth="3"><path d="M20 6 9 17l-5-5" /></svg>
              <span style={{ fontSize: 12.5, fontWeight: 600, color: "#2D4FC9" }}>분석 완료</span>
            </div>
          )}
          {elapsedSec !== null && (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", lineHeight: 1.3 }}>
              <span style={{ fontSize: 11.5, color: "#8A93A6" }}>처리 시간</span>
              <span style={{ fontSize: 12.5, fontWeight: 600, fontFamily: "'IBM Plex Mono', monospace" }}>{elapsedSec}s</span>
            </div>
          )}
          {(pageStatus === "done" || pageStatus === "error") && (
            <button
              onClick={reset}
              style={{ display: "flex", alignItems: "center", gap: 6, padding: "6px 13px", background: "transparent", border: "1px solid #E6E9EF", borderRadius: 8, cursor: "pointer", fontSize: 12.5, fontWeight: 600, color: "#5B6577", fontFamily: "inherit" }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 1 0 .49-3.55" /></svg>
              초기화
            </button>
          )}
        </div>
      </header>

      {/* ── Pipeline strip (shown during analyzing / done) ── */}
      {(pageStatus === "analyzing" || pageStatus === "done") && (
        <PipelineStrip
          steps={pipelineSteps}
          summaryNode={isDashboard ? (
            <DonutSummary ok={okCount} review={reviewCount} total={totalFields} />
          ) : undefined}
        />
      )}

      {/* ── Main content area ── */}
      {pageStatus === "idle" || pageStatus === "uploading" || pageStatus === "error" ? (
        <UploadZone pageStatus={pageStatus} fileName={fileName} errorMsg={errorMsg} onFileChange={handleFileChange} />
      ) : pageStatus === "analyzing" ? (
        <AnalyzingView progressSteps={progressSteps} pipeline={pipelineSteps} />
      ) : isDashboard ? (
        <ExtractionDashboard result={result!} fileName={fileName ?? ""} />
      ) : (
        /* Fallback for done but no structured data */
        <div style={{ flex: 1, overflowY: "auto", padding: 24 }}>
          <div style={{ maxWidth: 640, margin: "0 auto", background: "#fff", borderRadius: 14, border: "1px solid #F0D8AE", padding: 24 }}>
            <p style={{ fontSize: 13, color: "#B5701A", marginBottom: 12 }}>⚠️ 구조화 데이터를 추출하지 못했습니다.</p>
            <pre style={{ maxHeight: 480, overflow: "auto", background: "#1A1F29", color: "#E5E9F0", borderRadius: 10, padding: 16, fontSize: 12, lineHeight: 1.7 }}>
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  )
}
