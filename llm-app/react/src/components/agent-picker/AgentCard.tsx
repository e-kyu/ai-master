import { useNavigate } from "react-router-dom"
import { useAppState } from "../../context/appStateStore"
import type { Agent } from "../../types"
import { ArrowRightIcon, ChatBubbleIcon, DocumentScanIcon, SparkleIcon } from "../common/Icons"

const GRADIENTS = [
  "from-violet-500 to-indigo-600",
  "from-sky-500 to-blue-600",
  "from-fuchsia-500 to-purple-600",
  "from-rose-500 to-pink-600",
]

const MODE_ICONS: Record<string, typeof SparkleIcon> = {
  PpsAssistAgent: ChatBubbleIcon,
  NoticeScanAgent: DocumentScanIcon,
}

export function AgentCard({ agent, index }: { agent: Agent; index: number }) {
  const navigate = useNavigate()
  const { startNewConversation } = useAppState()
  const gradient = GRADIENTS[index % GRADIENTS.length]
  const Icon = MODE_ICONS[agent.mode] ?? SparkleIcon

  const handleStart = () => {
    startNewConversation(agent)
    navigate(`/agent/${agent.mode}`)
  }

  return (
    <div className="group relative flex flex-col overflow-hidden rounded-3xl border border-zinc-200/80 bg-white shadow-[0_1px_2px_rgba(0,0,0,0.04)] transition-all duration-300 hover:-translate-y-1.5 hover:border-transparent hover:shadow-2xl hover:shadow-zinc-300/60">
      <div className={`relative overflow-hidden bg-gradient-to-br ${gradient} px-6 py-7 text-white`}>
        <div className="absolute -right-8 -top-10 size-32 rounded-full bg-white/10 blur-2xl transition-transform duration-500 group-hover:scale-125" />
        <div className="absolute -bottom-12 -left-6 size-28 rounded-full bg-black/10 blur-2xl" />
        <div className="relative flex size-11 items-center justify-center rounded-2xl bg-white/15 ring-1 ring-white/25 backdrop-blur-sm">
          <Icon className="size-6" />
        </div>
        <h3 className="relative mt-4 text-lg font-bold tracking-tight">{agent.name}</h3>
        <p className="relative mt-2 line-clamp-3 text-sm leading-relaxed text-white/85">{agent.description}</p>
      </div>
      <div className="flex flex-1 items-end p-4">
        <button
          type="button"
          onClick={handleStart}
          className="flex w-full items-center justify-center gap-1.5 rounded-xl bg-zinc-900 px-4 py-2.5 text-sm font-semibold text-white transition-all duration-200 hover:gap-2.5 hover:bg-zinc-800"
        >
          시작하기
          <ArrowRightIcon className="size-4 transition-transform group-hover:translate-x-0.5" />
        </button>
      </div>
    </div>
  )
}
