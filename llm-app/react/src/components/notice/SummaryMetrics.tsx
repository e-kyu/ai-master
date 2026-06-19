interface Props {
  noticeType: string
  contractMethod: string
  bidEndDate: string
  itemCount: number
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-white px-4 py-3">
      <p className="text-xs text-zinc-400">{label}</p>
      <p className="mt-1 truncate text-base font-semibold text-zinc-900">{value}</p>
    </div>
  )
}

export function SummaryMetrics({ noticeType, contractMethod, bidEndDate, itemCount }: Props) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <MetricCard label="공고 종류" value={noticeType || "-"} />
      <MetricCard label="계약 방법" value={contractMethod || "-"} />
      <MetricCard label="입찰 마감" value={bidEndDate || "-"} />
      <MetricCard label="품목 수" value={String(itemCount)} />
    </div>
  )
}
