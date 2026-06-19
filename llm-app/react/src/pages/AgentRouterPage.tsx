import { useEffect } from "react"
import { useParams } from "react-router-dom"
import { useAppState } from "../context/appStateStore"
import { useAgents } from "../hooks/useAgents"
import { Spinner } from "../components/common/Spinner"
import { EmptyState } from "../components/common/EmptyState"
import { PpsAssistAgentPage } from "./PpsAssistAgentPage"
import { NoticeScanAgentPage } from "./NoticeScanAgentPage"

// 새로고침이나 직접 URL 진입처럼 selectedAgent 컨텍스트가 비어있는 경우를 위해
// 에이전트 목록에서 mode에 해당하는 메타데이터를 조회해 채워준다.
export function AgentRouterPage() {
  const { mode } = useParams<{ mode: string }>()
  const { selectedAgent, startNewConversation } = useAppState()
  const { agents, loading, error } = useAgents()

  const needsResolve = !selectedAgent || selectedAgent.mode !== mode

  useEffect(() => {
    if (!needsResolve || loading || !mode) return
    const match = agents.find((a) => a.mode === mode)
    if (match) startNewConversation(match)
  }, [needsResolve, loading, agents, mode, startNewConversation])

  if (needsResolve) {
    if (loading) {
      return (
        <div className="flex items-center justify-center gap-2 py-16 text-zinc-400">
          <Spinner className="size-5" /> 에이전트 정보를 불러오는 중...
        </div>
      )
    }
    if (error || !agents.some((a) => a.mode === mode)) {
      return (
        <div className="mx-auto max-w-xl px-4 py-16">
          <EmptyState title="에이전트를 찾을 수 없습니다" description="목록으로 돌아가 다시 시도해주세요." />
        </div>
      )
    }
    return (
      <div className="flex items-center justify-center gap-2 py-16 text-zinc-400">
        <Spinner className="size-5" /> 준비 중...
      </div>
    )
  }

  if (mode === "PpsAssistAgent") return <PpsAssistAgentPage />
  if (mode === "NoticeScanAgent") return <NoticeScanAgentPage />

  return (
    <div className="mx-auto max-w-xl px-4 py-16">
      <EmptyState title="지원하지 않는 에이전트 모드입니다" description={mode} />
    </div>
  )
}
