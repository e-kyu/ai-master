import { useEffect, useRef, useState } from "react"
import { useAppState } from "../context/appStateStore"
import { uploadFile } from "../api/upload"
import { analyzeNotice } from "../api/qna"
import { UploadIcon, RefreshIcon } from "../components/common/Icons"
import { Spinner } from "../components/common/Spinner"
import { SummaryMetrics } from "../components/notice/SummaryMetrics"
import { ResultTabs } from "../components/notice/ResultTabs"
import type { NoticeScanResult } from "../types"

type Status = "idle" | "uploading" | "analyzing" | "done" | "error"

export function NoticeScanAgentPage() {
  const { selectedAgent, convrstnId, messages, refreshHistory } = useAppState()
  const [status, setStatus] = useState<Status>("idle")
  const [fileName, setFileName] = useState<string | null>(null)
  const [result, setResult] = useState<NoticeScanResult | null>(null)
  const [statusMessage, setStatusMessage] = useState("파일을 선택하면 자동으로 분석을 시작합니다.")
  const fileInputRef = useRef<HTMLInputElement>(null)

  // convrstnId 변경 시(신규 또는 이력 복원) 상태 동기화
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (messages.length === 0) {
      setStatus("idle")
      setFileName(null)
      setResult(null)
      setStatusMessage("파일을 선택하면 자동으로 분석을 시작합니다.")
      return
    }
    const last = messages[messages.length - 1]
    if (!last.answer) return
    try {
      const parsed = JSON.parse(last.answer) as NoticeScanResult
      const name = last.question.split(/[/\\]/).pop() ?? last.question
      setFileName(name)
      setResult(parsed)
      setStatus("done")
      setStatusMessage("이전 분석 결과를 불러왔습니다.")
    } catch {
      // answer가 JSON이 아닌 경우(PpsAssist 이력 등) 무시
    }
  }, [convrstnId]) // messages는 convrstnId와 함께 원자적으로 변경되므로 의존성 제외

  if (!selectedAgent) return null

  const reset = () => {
    setStatus("idle")
    setFileName(null)
    setResult(null)
    setStatusMessage("파일을 선택하면 자동으로 분석을 시작합니다.")
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ""
    if (!file) return

    setFileName(file.name)
    setResult(null)
    setStatus("uploading")
    setStatusMessage("파일을 저장하고 분석을 준비 중입니다...")

    try {
      const uploaded = await uploadFile(file)

      setStatus("analyzing")
      setStatusMessage("⚙️ 파일을 분석하는 중입니다...")

      const res = await analyzeNotice({
        agent_id: selectedAgent.agent_id,
        agent_mode: selectedAgent.mode,
        convrstnId,
        fileFullPath: uploaded.fileFullPath,
        question: "",
        enableExtDocse: false,
      })

      setResult(res)
      setStatus("done")
      setStatusMessage("분석 결과가 아래에 표시됩니다.")
      refreshHistory()
    } catch {
      setStatus("error")
      setStatusMessage("API 요청 오류: 분석에 실패했습니다.")
    }
  }

  const data = result?.extracted_data
  const hasData = !!data && (data.general || data.execution || (data.items && data.items.length > 0))

  return (
    <div className="mx-auto max-w-4xl px-4 py-6 sm:px-6 lg:py-8">
      <p className="mt-0.5 text-sm text-zinc-500">{selectedAgent.description}</p>

      <div className="mt-5 flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <p className="text-sm font-semibold text-zinc-700">📄 분석 대상 문서</p>
          {fileName && (
            <button
              type="button"
              onClick={reset}
              className="flex items-center gap-1 rounded-md px-2 py-1 text-xs text-zinc-500 hover:bg-zinc-100"
            >
              <RefreshIcon className="size-3.5" /> 초기화
            </button>
          )}
        </div>

        <label
          className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed px-4 py-10 text-center transition ${
            status === "uploading" || status === "analyzing"
              ? "border-brand-300 bg-brand-50/50"
              : "border-zinc-200 hover:border-brand-300 hover:bg-brand-50/30"
          }`}
        >
          <input ref={fileInputRef} type="file" accept=".pdf,.html,.htm" className="hidden" onChange={handleFileChange} />
          {status === "uploading" || status === "analyzing" ? (
            <Spinner className="size-6 text-brand-500" />
          ) : (
            <UploadIcon className="size-7 text-zinc-400" />
          )}
          <span className="text-sm font-medium text-zinc-600">
            공고문 파일(PDF, HTML)을 업로드하세요
          </span>
          <span className="text-xs text-zinc-400">클릭하여 파일 선택</span>
        </label>

        <div className="rounded-xl bg-zinc-50 px-4 py-3 text-sm text-zinc-600">
          {fileName ? (
            <p>
              📁 <span className="font-medium">{fileName}</span>
              <br />
              💡 {statusMessage}
            </p>
          ) : (
            <p>ℹ️ {statusMessage}</p>
          )}
        </div>

        {status === "done" && result && (
          <div className="flex flex-col gap-4">
            {!hasData ? (
              <div>
                <p className="mb-2 text-sm text-amber-600">⚠️ 추출된 구조화 데이터가 없습니다.</p>
                <pre className="max-h-[480px] overflow-auto rounded-xl bg-zinc-900 p-4 text-xs text-zinc-100">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </div>
            ) : (
              <>
                {result.is_violating && (
                  <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700">
                    ⚠️ <span className="font-medium">규정 위반 의심:</span> 공고문 내 독소조항이나 규정 위반 가능성이
                    감지되었습니다.
                  </div>
                )}

                <SummaryMetrics
                  noticeType={data?.general?.noticeType ?? ""}
                  contractMethod={data?.general?.contractMethod ?? ""}
                  bidEndDate={(data?.execution?.bidEndDate ?? "").split("T")[0]}
                  itemCount={data?.items?.length ?? 0}
                />

                <ResultTabs
                  general={data?.general ?? {}}
                  execution={data?.execution ?? {}}
                  items={data?.items ?? []}
                  data={data}
                  raw={result}
                />
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
