import { useEffect, useRef, useState } from "react"
import { useAppState } from "../context/appStateStore"
import { askQuestionStreaming } from "../api/qna"
import { ChatMessageBubble } from "../components/chat/ChatMessage"
import { ChatInput } from "../components/chat/ChatInput"
import { EmptyState } from "../components/common/EmptyState"

export function PpsAssistAgentPage() {
  const {
    selectedAgent,
    convrstnId,
    messages,
    enableExtDocse,
    setEnableExtDocse,
    appendMessage,
    updateLastMessageAnswer,
    refreshHistory,
  } = useAppState()

  const [sending, setSending] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, sending])

  if (!selectedAgent) return null

  const handleSend = async (question: string, fileFullPath: string) => {
    appendMessage({ question, answer: "" })
    setSending(true)
    try {
      await askQuestionStreaming(
        {
          agent_id: selectedAgent.agent_id,
          agent_mode: selectedAgent.mode,
          convrstnId,
          fileFullPath,
          question,
          enableExtDocse,
        },
        (_chunk, fullTextSoFar) => updateLastMessageAnswer(fullTextSoFar),
      )
      refreshHistory()
    } catch {
      updateLastMessageAnswer("죄송합니다. 답변을 생성하는 중 오류가 발생했습니다.")
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col">
      <div className="border-b border-zinc-100 px-4 py-4 sm:px-6">
        <p className="mt-0.5 text-sm text-zinc-500">{selectedAgent.description}</p>
        <label className="mt-3 flex items-center gap-2 text-xs text-zinc-500">
          <input
            type="checkbox"
            checked={enableExtDocse}
            onChange={(e) => setEnableExtDocse(e.target.checked)}
            className="size-3.5 rounded border-zinc-300 text-brand-600 focus:ring-brand-400"
          />
          외부 문서 검색 허용
        </label>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 sm:px-6">
        {messages.length === 0 ? (
          <EmptyState title="대화를 시작해보세요" description="궁금한 내용을 입력하면 답변을 받아볼 수 있습니다." />
        ) : (
          <div className="flex flex-col gap-5">
            {messages.map((m, i) => (
              <ChatMessageBubble key={i} message={m} streaming={sending && i === messages.length - 1} />
            ))}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <ChatInput disabled={sending} onSend={handleSend} />
    </div>
  )
}
