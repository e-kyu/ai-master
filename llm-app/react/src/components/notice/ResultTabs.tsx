import { useState } from "react"
import ReactMarkdown from "react-markdown"
import { ItemsTable } from "./ItemsTable"
import type { NoticeExecutionInfo, NoticeGeneralInfo, NoticeItem, NoticeScanResult } from "../../types"

interface Props {
  general: NoticeGeneralInfo
  execution: NoticeExecutionInfo
  items: NoticeItem[]
  data: unknown
  raw: NoticeScanResult
}

const GENERAL_LABELS: [keyof NoticeGeneralInfo, string][] = [
  ["noticeNo", "공고번호"],
  ["refNo", "참조번호"],
  ["postDate", "게시일시"],
  ["agency", "공고기관"],
  ["demandAgency", "수요기관"],
  ["noticeType", "공고 종류"],
  ["contractType", "계약 종류"],
  ["contractForm", "계약 형태"],
  ["bidMethod", "입찰 방법"],
  ["stockType", "비축 구분"],
  ["rebidYn", "재입찰 여부"],
]

const EXECUTION_LABELS: [keyof NoticeExecutionInfo, string][] = [
  ["manager", "담당자"],
  ["bidStartDate", "입찰 시작일시"],
  ["bidEndDate", "입찰 마감일시"],
  ["openDate", "개찰일시"],
  ["openPlace", "개찰장소"],
  ["depositExemptYn", "보증금 면제"],
  ["depositDate", "보증금 납부일"],
  ["relatedNotice", "관련 공고"],
]

function Field({ label, value }: { label: string; value?: unknown }) {
  const display = value == null || value === "" ? "-" : String(value)
  return (
    <div>
      <p className="text-xs text-zinc-400">{label}</p>
      <p className="mt-0.5 text-sm text-zinc-800">{display}</p>
    </div>
  )
}

function extraEntries<T extends Record<string, unknown>>(obj: T, known: (keyof T)[]) {
  return Object.entries(obj).filter(([key]) => !known.includes(key as keyof T) && key !== "noticeName" && key !== "awardMethod" && key !== "awardDetail")
}

export function ResultTabs({ general, execution, items, data, raw }: Props) {
  const hasProgress = (raw.extracted_data?.progresses?.length ?? 0) > 0 || (raw.extracted_data?.statuses?.length ?? 0) > 0

  const tabs = ["상세정보", "품목", ...(hasProgress ? ["진행현황"] : []), "원문", "자동입력 항목"] as const
  type Tab = (typeof tabs)[number]
  const [active, setActive] = useState<Tab>("상세정보")
  const [copyStatus, setCopyStatus] = useState<string | null>(null)

  const copyJsonToClipboard = async () => {
    const text = JSON.stringify(data, null, 2)

    try {
      await navigator.clipboard.writeText(text)
      setCopyStatus("복사되었습니다.")
    } catch(e) {
      console.error("Failed to copy JSON to clipboard:", e)
      setCopyStatus("복사에 실패했습니다.")
    }

    window.setTimeout(() => setCopyStatus(null), 2000)
  }

  const generalExtras = extraEntries(general, GENERAL_LABELS.map(([key]) => key))
  const executionExtras = extraEntries(execution, EXECUTION_LABELS.map(([key]) => key))

  return (
    <div>
      <div className="flex gap-1 overflow-x-auto border-b border-zinc-200">
        {tabs.map((tab) => (
          <button
            key={tab}
            type="button"
            onClick={() => setActive(tab)}
            className={`shrink-0 border-b-2 px-3.5 py-2 text-sm font-medium transition ${
              active === tab
                ? "border-brand-600 text-brand-700"
                : "border-transparent text-zinc-500 hover:text-zinc-700"
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      <div className="py-4">
        {active === "상세정보" && (
          <div className="flex flex-col gap-4">
            <h3 className="text-base font-semibold text-zinc-900">{general.noticeName || "공고명 없음"}</h3>

            <div className="rounded-xl border border-zinc-200 p-4">
              <p className="mb-2 text-xs font-medium text-zinc-500">🎯 낙찰 방법 및 상세</p>
              <p className="text-sm text-zinc-700">
                <span className="font-medium">방법:</span> {general.awardMethod || "-"}
              </p>
              <p className="mt-1 text-sm text-zinc-700">
                <span className="font-medium">상세:</span> {general.awardDetail || "-"}
              </p>
            </div>

            <div>
              <p className="mb-2 text-xs font-medium text-zinc-500">📋 공고 기본정보</p>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {GENERAL_LABELS.map(([key, label]) => (
                  <Field key={key} label={label} value={general[key]} />
                ))}
              </div>
            </div>

            <div>
              <p className="mb-2 text-xs font-medium text-zinc-500">🗓️ 입찰 일정 및 장소</p>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {EXECUTION_LABELS.map(([key, label]) => (
                  <Field key={key} label={label} value={execution[key]} />
                ))}
              </div>
            </div>

            {(generalExtras.length > 0 || executionExtras.length > 0) && (
              <div>
                <p className="mb-2 text-xs font-medium text-zinc-500">➕ 추가 정보</p>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  {[...generalExtras, ...executionExtras].map(([key, value]) => (
                    <Field key={key} label={key} value={value} />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {active === "품목" && <ItemsTable items={items} />}

        {active === "진행현황" && (
          <div className="flex flex-col gap-4">
            {(raw.extracted_data?.statuses?.length ?? 0) > 0 && (
              <div>
                <p className="mb-2 text-xs font-medium text-zinc-500">상태 변경 이력</p>
                <pre className="max-h-[320px] overflow-auto rounded-xl bg-zinc-900 p-4 text-xs text-zinc-100">
                  {JSON.stringify(raw.extracted_data?.statuses, null, 2)}
                </pre>
              </div>
            )}
            {(raw.extracted_data?.progresses?.length ?? 0) > 0 && (
              <div>
                <p className="mb-2 text-xs font-medium text-zinc-500">진행 단계</p>
                <pre className="max-h-[320px] overflow-auto rounded-xl bg-zinc-900 p-4 text-xs text-zinc-100">
                  {JSON.stringify(raw.extracted_data?.progresses, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}

        {active === "원문" && (
          <div className="max-h-[480px] overflow-auto rounded-xl bg-zinc-50 p-4 text-sm text-zinc-700">
            <ReactMarkdown>{raw.document_text || "원문 텍스트가 없습니다."}</ReactMarkdown>
          </div>
        )}

        {active === "자동입력 항목" && (
          <div>
            
            <div className="mb-2 flex items-center justify-between">
              <p className="text-xs font-medium text-zinc-500">추출 데이터 구조</p>
              <button type="button" onClick={copyJsonToClipboard} className="rounded bg-blue-600 px-3 py-1 text-xs font-medium text-white hover:bg-blue-700">
                {copyStatus || "클립보드에 복사"}
              </button>
            </div>
            
            <pre className="max-h-[480px] overflow-auto rounded-xl bg-zinc-900 p-4 text-xs text-zinc-100">
              {JSON.stringify(data, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  )
}
