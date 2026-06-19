import { useNavigate } from "react-router-dom"
import { useAppState } from "../../context/appStateStore"
import type { Agent } from "../../types"
import { ChatBubbleIcon } from "../../components/common/Icons"

const GRADIENTS = [
  "from-violet-500 to-indigo-600",
  "from-sky-500 to-blue-600",
  "from-fuchsia-500 to-purple-600",
  "from-rose-500 to-pink-600",
]

export function AgentCard({ agent, index }: { agent: Agent; index: number }) {
  const navigate = useNavigate()
  const { startNewConversation } = useAppState()
  const gradient = GRADIENTS[index % GRADIENTS.length]

  const handleStart = () => {
    startNewConversation(agent)
    navigate(`/agent/${agent.mode}`)
  }

  return (
    <div className={`flex flex-col overflow-hidden rounded-2xl border border-transparent bg-gradient-to-br ${gradient} shadow-sm transition hover:-translate-y-0.5 hover:shadow-md`}>
      <div className="px-5 py-5 text-white">
        <h3 className="text-base font-semibold">{agent.name}</h3>
        <p className="mt-1.5 line-clamp-3 text-sm text-white/85">{agent.description}</p>
      </div>
      <div className="flex flex-1 items-end p-4">

        
        <button
          type="button"
          onClick={handleStart}
          className="w-full rounded-lg px-3 py-2 text-sm font-medium text-white transition hover:opacity-90"
        >
            <div className="inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-indigo-600 to-sky-500 px-8 py-2 text-sm font-semibold text-white shadow-md">
              <ChatBubbleIcon className="size-4" />
              시작하기
            </div>
        </button>
      </div>
    </div>
  )
}
