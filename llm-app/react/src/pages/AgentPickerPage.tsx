import { useAgents } from "../hooks/useAgents"
import { AgentCard } from "../components/agent-picker/AgentCard"
import { Spinner } from "../components/common/Spinner"
import { EmptyState } from "../components/common/EmptyState"
import { SparkleIcon } from "../components/common/Icons"

export function AgentPickerPage() {
  const { agents, loading, error } = useAgents()

  return (
    <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6 lg:px-8">
      <div className="relative mb-12 overflow-hidden rounded-3xl bg-gradient-to-br from-brand-600 via-brand-500 to-violet-500 px-6 py-10 text-white shadow-lg shadow-brand-200 sm:px-10 sm:py-14">
        <div className="absolute -right-16 -top-16 size-64 rounded-full bg-white/10 blur-3xl" />
        <div className="absolute -bottom-20 left-1/3 size-72 rounded-full bg-black/10 blur-3xl" />

        <div className="relative inline-flex items-center gap-1.5 rounded-full bg-white/15 px-3 py-1 text-xs font-medium text-white/90 ring-1 ring-white/25 backdrop-blur-sm">
          <SparkleIcon className="size-3.5" />
          공공조달 어시스턴트
        </div>
        <h1 className="relative mt-4 text-base font-medium text-white/80 sm:text-lg">안녕하세요!</h1>
        <h2 className="relative mt-1 text-3xl font-extrabold tracking-tight sm:text-4xl">
          어떤 도움이 필요하신가요?
        </h2>
        <p className="relative mt-3 max-w-md text-sm text-white/80 sm:text-base">
          아래에서 원하는 에이전트를 선택하면 바로 대화를 시작할 수 있어요.
        </p>
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
