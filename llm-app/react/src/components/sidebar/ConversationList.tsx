import { useState } from "react"
import { useNavigate } from "react-router-dom"
import { fetchConversationDetail } from "../../api/conversations"
import { useAppState } from "../../context/appStateStore"
import { useConversationHistory } from "../../hooks/useConversationHistory"
import { Spinner } from "../common/Spinner"
import { EmptyState } from "../common/EmptyState"
import { ConversationItem } from "./ConversationItem"
import type { ConvrstnListItem } from "../../types"

export function ConversationList() {
  const navigate = useNavigate()
  const { resumeConversation, historyVersion } = useAppState()
  const { items, loading, error, remove } = useConversationHistory(historyVersion)
  const [resumingId, setResumingId] = useState<string | null>(null)

  const handleResume = async (item: ConvrstnListItem) => {
    if (!item.mode || !item.agent_id) return
    setResumingId(item.convrstn_id)
    try {
      const details = await fetchConversationDetail(item.convrstn_id)
      const messages = details.map((d) => ({ question: d.question, answer: d.answer ?? "" }))
      resumeConversation(
        {
          agent_id: item.agent_id,
          mode: item.mode,
          name: item.name ?? "",
          description: item.description ?? "",
          created_at: item.created_at,
        },
        item.convrstn_id,
        messages,
      )
      navigate(`/agent/${item.mode}`)
    } finally {
      setResumingId(null)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center gap-2 py-8 text-sm text-zinc-400">
        <Spinner className="size-4" /> 불러오는 중...
      </div>
    )
  }

  if (error) {
    return <EmptyState title={error} />
  }

  if (items.length === 0) {
    return <EmptyState title="대화 이력이 없습니다" description="에이전트를 선택하고 대화를 시작해보세요." />
  }

  return (
    <div className="flex flex-col gap-0.5">
      {items.map((item) => (
        <div key={item.convrstn_id} className="relative">
          <ConversationItem item={item} onResume={handleResume} onDelete={(i) => remove(i.convrstn_id)} />
          {resumingId === item.convrstn_id && (
            <div className="absolute inset-0 flex items-center justify-center rounded-lg bg-white/70">
              <Spinner className="size-4 text-brand-500" />
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
