import { useRef, useState } from "react"
import { uploadFile } from "../../api/upload"
import { PaperclipIcon, SendIcon, CloseIcon } from "../common/Icons"
import { Spinner } from "../common/Spinner"

interface Props {
  disabled?: boolean
  onSend: (question: string, fileFullPath: string) => void
}

export function ChatInput({ disabled, onSend }: Props) {
  const [text, setText] = useState("")
  const [attached, setAttached] = useState<{ name: string; fullPath: string } | null>(null)
  const [uploading, setUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ""
    if (!file) return
    setUploading(true)
    try {
      const res = await uploadFile(file)
      setAttached({ name: res.fileName, fullPath: res.fileFullPath })
    } catch {
      setAttached(null)
    } finally {
      setUploading(false)
    }
  }

  const handleSend = () => {
    const trimmed = text.trim()
    if (!trimmed || disabled) return
    onSend(trimmed, attached?.fullPath ?? "")
    setText("")
    setAttached(null)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="border-t border-zinc-200 bg-sky-50 p-3 sm:p-4">
      {attached && (
        <div className="mb-2 flex items-center gap-2 rounded-lg bg-zinc-100 px-3 py-1.5 text-xs text-zinc-600">
          <PaperclipIcon className="size-3.5 shrink-0" />
          <span className="min-w-0 flex-1 truncate">{attached.name}</span>
          <button type="button" onClick={() => setAttached(null)} className="shrink-0 text-zinc-400 hover:text-zinc-700">
            <CloseIcon className="size-3.5" />
          </button>
        </div>
      )}
      <div className="flex items-end gap-2">
        <input ref={fileInputRef} type="file" className="hidden" onChange={handleFileChange} />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled || uploading}
          className="flex size-10 shrink-0 items-center justify-center rounded-xl border border-zinc-200 text-zinc-500 transition hover:bg-zinc-50 disabled:opacity-50"
          aria-label="파일 첨부"
        >
          {uploading ? <Spinner className="size-4" /> : <PaperclipIcon className="size-4" />}
        </button>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          rows={1}
          placeholder="메세지를 입력하세요."
          className="max-h-32 flex-1 resize-none rounded-xl border border-zinc-200 px-3.5 py-2.5 text-sm text-zinc-800 outline-none placeholder:text-zinc-400 focus:border-brand-400 focus:ring-1 focus:ring-brand-400 disabled:bg-zinc-50"
        />
        <button
          type="button"
          onClick={handleSend}
          disabled={disabled || !text.trim()}
          className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-brand-600 text-white transition hover:bg-brand-700 disabled:opacity-40"
          aria-label="전송"
        >
          <SendIcon className="size-4" />
        </button>
      </div>
    </div>
  )
}
