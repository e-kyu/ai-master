import { useCallback, useEffect, useState } from "react"
import { deleteConversation, fetchConversationHistory } from "../api/conversations"
import type { ConvrstnListItem } from "../types"

export function useConversationHistory(refreshKey: number) {
  const [items, setItems] = useState<ConvrstnListItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    fetchConversationHistory()
      .then((data) => {
        if (!cancelled) {
          setItems(data)
          setError(null)
        }
      })
      .catch(() => {
        if (!cancelled) setError("대화 이력을 불러오지 못했습니다.")
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [refreshKey])

  const remove = useCallback(async (convrstnId: string) => {
    await deleteConversation(convrstnId)
    setItems((prev) => prev.filter((item) => item.convrstn_id !== convrstnId))
  }, [])

  return { items, loading, error, remove }
}
