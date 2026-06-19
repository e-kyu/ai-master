import { useState } from "react"
import { ItemsTable } from "./ItemsTable"
import type { NoticeExecutionInfo, NoticeGeneralInfo, NoticeItem, NoticeScanResult } from "../../types"

const TABS = ["상세정보", "품목", "JSON", "원문"] as const
type Tab = (typeof TABS)[number]

interface Props {
  general: NoticeGeneralInfo
  execution: NoticeExecutionInfo
  items: NoticeItem[]
  data: unknown
  raw: NoticeScanResult
}

function Field({ label, value }: { label: string; value?: string | null }) {
  return (
    <div>
      <p className="text-xs text-zinc-400">{label}</p>
      <p className="mt-0.5 text-sm text-zinc-800">{value || "-"}</p>
    </div>
  )
}

export function ResultTabs({ general, execution, items, data, raw }: Props) {
  const [active, setActive] = useState<Tab>("상세정보")

  return (
    <div>
      <div className="flex gap-1 overflow-x-auto border-b border-zinc-200">
        {TABS.map((tab) => (
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

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="flex flex-col gap-3">
                <Field label="공고번호" value={general.noticeNo as string} />
                <Field label="게시일시" value={general.postDate as string} />
                <Field label="공고기관" value={general.agency as string} />
              </div>
              <div className="flex flex-col gap-3">
                <Field label="담당자" value={execution.manager as string} />
                <Field label="개찰일시" value={execution.openDate as string} />
                <Field label="보증금면제" value={execution.depositExemptYn as string} />
              </div>
            </div>
            <Field label="개찰장소" value={execution.openPlace as string} />
          </div>
        )}

        {active === "품목" && <ItemsTable items={items} />}

        {active === "JSON" && (
          <div>
            <p className="mb-2 text-xs font-medium text-zinc-500">추출 데이터 구조</p>
            <pre className="max-h-[480px] overflow-auto rounded-xl bg-zinc-900 p-4 text-xs text-zinc-100">
              {JSON.stringify(data, null, 2)}
            </pre>
          </div>
        )}

        {active === "원문" && (
          <pre className="max-h-[480px] overflow-auto rounded-xl bg-zinc-900 p-4 text-xs text-zinc-100">
            {JSON.stringify(raw, null, 2)}
          </pre>
        )}
      </div>
    </div>
  )
}
