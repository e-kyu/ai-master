interface Props {
  noticeType: string
  contractMethod: string
  bidEndDate: string
  itemCount: number
}

function MetricCard({
  label,
  value,
  tone = "default",
}: {
  label: string
  value: string
  tone?: "default" | "warning" | "danger"
}) {
  const toneClass =
    tone === "danger"
      ? "border-red-200 bg-red-50 text-red-700"
      : tone === "warning"
        ? "border-amber-200 bg-amber-50 text-amber-700"
        : "border-zinc-200 bg-white text-zinc-900"

  return (
    <div className={`rounded-xl border px-4 py-3 ${toneClass}`}>
      <p className="text-xs text-zinc-400">{label}</p>
      <p className="mt-1 truncate text-base font-semibold">{value}</p>
    </div>
  )
}

function describeDeadline(bidEndDate: string): { label: string; value: string; tone: "default" | "warning" | "danger" } {
  if (!bidEndDate) return { label: "입찰 마감", value: "-", tone: "default" }

  const deadline = new Date(bidEndDate)
  if (Number.isNaN(deadline.getTime())) return { label: "입찰 마감", value: bidEndDate, tone: "default" }

  const today = new Date()
  today.setHours(0, 0, 0, 0)
  deadline.setHours(0, 0, 0, 0)
  const daysLeft = Math.round((deadline.getTime() - today.getTime()) / (1000 * 60 * 60 * 24))

  if (daysLeft < 0) return { label: "입찰 마감", value: `${bidEndDate} (마감)`, tone: "default" }
  if (daysLeft === 0) return { label: "입찰 마감", value: `${bidEndDate} (D-Day)`, tone: "danger" }
  if (daysLeft <= 3) return { label: "입찰 마감", value: `${bidEndDate} (D-${daysLeft})`, tone: "danger" }
  if (daysLeft <= 7) return { label: "입찰 마감", value: `${bidEndDate} (D-${daysLeft})`, tone: "warning" }
  return { label: "입찰 마감", value: `${bidEndDate} (D-${daysLeft})`, tone: "default" }
}

export function SummaryMetrics({ noticeType, contractMethod, bidEndDate, itemCount }: Props) {
  const deadline = describeDeadline(bidEndDate)

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <MetricCard label="공고 종류" value={noticeType || "-"} />
      <MetricCard label="계약 방법" value={contractMethod || "-"} />
      <MetricCard label={deadline.label} value={deadline.value} tone={deadline.tone} />
      <MetricCard label="품목 수" value={String(itemCount)} />
    </div>
  )
}
