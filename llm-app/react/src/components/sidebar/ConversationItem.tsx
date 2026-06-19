import { useState } from "react"
import { ChatBubbleIcon, TrashIcon } from "../common/Icons"
import { Spinner } from "../common/Spinner"
import type { ConvrstnListItem } from "../../types"

interface Props {
  item: ConvrstnListItem
  onResume: (item: ConvrstnListItem) => void
  onDelete: (item: ConvrstnListItem) => Promise<void>
}

export function ConversationItem({ item, onResume, onDelete }: Props) {
  const [deleting, setDeleting] = useState(false)
  const topic = item.topic?.trim() || "(제목 없음)"
  const displayTopic = topic.length > 26 ? `${topic.slice(0, 26)}···` : topic
  const displayDate = item.created_at ? item.created_at.slice(0, 16).replace("T", " ") : ""

  const handleDelete = async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (deleting) return
    setDeleting(true)
    try {
      await onDelete(item)
    } finally {
      setDeleting(false)
    }
  }

  return (
    <button
      type="button"
      onClick={() => onResume(item)}
      className="group flex w-full items-start gap-2.5 rounded-lg px-3 py-2.5 text-left transition hover:bg-brand-50"
    >
      <ChatBubbleIcon className="mt-0.5 size-4 shrink-0 text-zinc-400 group-hover:text-brand-500" />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-medium text-zinc-700 group-hover:text-zinc-900">
          {displayTopic}
        </span>
        <span className="mt-0.5 flex items-center gap-1.5 text-[11px] text-zinc-400">
          {item.name && <span className="truncate">{item.name}</span>}
          {item.name && displayDate && <span>·</span>}
          <span>{displayDate}</span>
        </span>
      </span>
      <span
        role="button"
        aria-label="대화 삭제"
        onClick={handleDelete}
        className="shrink-0 rounded-md p-1.5 text-zinc-300 opacity-0 transition hover:bg-red-50 hover:text-red-500 group-hover:opacity-100"
      >
        {deleting ? <Spinner className="size-3.5" /> : <TrashIcon className="size-3.5" />}
      </span>
    </button>
  )
}
