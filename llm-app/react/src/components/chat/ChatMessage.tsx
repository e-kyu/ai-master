import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import type { ChatMessage as ChatMessageType } from "../../types"

export function ChatMessageBubble({ message, streaming }: { message: ChatMessageType; streaming?: boolean }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-brand-600 px-4 py-2.5 text-sm text-white shadow-sm sm:max-w-[70%]">
          {message.question}
        </div>
      </div>

      {(message.answer || streaming) && (
        <div className="flex justify-start">
          <div className="max-w-[90%] rounded-2xl rounded-bl-md border border-zinc-200 bg-white px-4 py-3 text-sm text-zinc-800 shadow-sm sm:max-w-[80%]">
            <div className="prose prose-sm prose-zinc max-w-none prose-p:my-1.5 prose-headings:my-2">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.answer || ""}</ReactMarkdown>
            </div>
            {streaming && <span className="ml-0.5 inline-block w-1.5 animate-pulse text-brand-500">▌</span>}
          </div>
        </div>
      )}
    </div>
  )
}
