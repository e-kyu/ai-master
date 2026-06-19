import { useAgents } from "../hooks/useAgents"
import { AgentCard } from "../components/agent-picker/AgentCard"
import { Spinner } from "../components/common/Spinner"
import { EmptyState } from "../components/common/EmptyState"

export function AgentPickerPage() {
  const { agents, loading, error } = useAgents()

  return (
    <div className="mx-auto max-w-6xl px-4 py-1 sm:px-6 lg:px-8">

      <div className="mx-auto max-w-2xl text-left">

        <h1 className="mt-5 text-lg font-medium text-zinc-500">안녕하세요!</h1>
        <h2 className="mt-2 mb-6 text-2xl font-extrabold tracking-tight sm:text-3xl">어떤 도움이 필요하신가요?</h2>
        
      </div>



      {loading && (
        <div className="flex items-center justify-center gap-2 py-20 text-zinc-400">
          <Spinner className="size-5" /> 에이전트 목록을 불러오는 중...
        </div>
      )}

      {!loading && error && <EmptyState title={error} />}

      {!loading && !error && agents.length === 0 && (
        <EmptyState title="등록된 에이전트가 없습니다" description="잠시 후 다시 시도해주세요." />
      )}

      {!loading && !error && agents.length > 0 && (
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {agents.map((agent, i) => (
            <AgentCard key={agent.agent_id} agent={agent} index={i} />
          ))}
        </div>
      )}

      <p className="mt-10 text-center text-sm text-zinc-400">
        원하는 에이전트의 <span className="font-medium text-zinc-500">'시작하기'</span> 버튼을 클릭하면 대화
        화면으로 이동합니다.
      </p>
    </div>
  )
}
