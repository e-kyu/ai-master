import { useEffect, useState } from "react"
import { fetchAgentList } from "../api/agents"
import type { Agent } from "../types"

export function useAgents() {
  const [agents, setAgents] = useState<Agent[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    fetchAgentList()
      .then((data) => {
        if (!cancelled) setAgents(data)
      })
      .catch(() => {
        if (!cancelled) setError("에이전트 목록을 불러오지 못했습니다.")
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  return { agents, loading, error }
}
