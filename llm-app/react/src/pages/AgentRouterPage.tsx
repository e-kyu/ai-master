import { useEffect, useState } from "react"
import { useParams, useSearchParams } from "react-router-dom"
import { useAppState } from "../context/appStateStore"
import { useAgents } from "../hooks/useAgents"
import { fetchConversationDetail } from "../api/conversations"
import { Spinner } from "../components/common/Spinner"
import { EmptyState } from "../components/common/EmptyState"
import { PpsAssistAgentPage } from "./PpsAssistAgentPage"
import { NoticeScanAgentPage } from "./NoticeScanAgentPage"

// 새로고침이나 직접 URL 진입처럼 selectedAgent 컨텍스트가 비어있는 경우를 위해
// 에이전트 목록에서 mode에 해당하는 메타데이터를 조회해 채워준다.
// URL에 ?convrstnId=<id>가 붙어 있으면(예: naraWeb-demo의 "AI 자동입력" 팝업에서 대화 이력 클릭)
// 새 대화를 시작하는 대신 그 대화 이력을 그대로 불러와 이어서 보여준다.
export function AgentRouterPage() {
  const { mode } = useParams<{ mode: string }>()
  const [searchParams] = useSearchParams()
  const requestedConvrstnId = searchParams.get("convrstnId")
  const { selectedAgent, convrstnId, startNewConversation, resumeConversation } = useAppState()
  const { agents, loading, error } = useAgents()
  const [resolvingHistory, setResolvingHistory] = useState(false)

  const needsHistoryResolve = !!requestedConvrstnId && convrstnId !== requestedConvrstnId
  const needsResolve = !selectedAgent || selectedAgent.mode !== mode || needsHistoryResolve

  useEffect(() => {
    if (!needsResolve || loading || !mode) return
    const match = agents.find((a) => a.mode === mode)
    if (!match) return

    if (requestedConvrstnId) {
      setResolvingHistory(true)
      fetchConversationDetail(requestedConvrstnId)
        .then((details) => {
          const messages = details.map((d) => ({ question: d.question, answer: d.answer ?? "" }))
          resumeConversation(match, requestedConvrstnId, messages)
        })
        .finally(() => setResolvingHistory(false))
    } else {
      startNewConversation(match)
    }
  }, [needsResolve, loading, agents, mode, requestedConvrstnId, startNewConversation, resumeConversation])

  if (needsResolve) {
    if (loading || resolvingHistory) {
      return (
        <div className="flex items-center justify-center gap-2 py-16 text-zinc-400">
          <Spinner className="size-5" /> {resolvingHistory ? "대화 이력을 불러오는 중..." : "에이전트 정보를 불러오는 중..."}
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
