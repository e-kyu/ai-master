import { useEffect, useRef, useState } from "react"
import { useNavigate } from "react-router-dom"
import { useAppState } from "../../context/appStateStore"
import { CloseIcon, PlusIcon } from "../common/Icons"
import { ConversationList } from "./ConversationList"

const MIN_WIDTH = 240
const MAX_WIDTH = 480
const DEFAULT_WIDTH = 350

export function Sidebar() {
  const navigate = useNavigate()
  const { sidebarOpen, setSidebarOpen } = useAppState()
  const [width, setWidth] = useState(DEFAULT_WIDTH)
  const [isResizing, setIsResizing] = useState(false)
  const draggingRef = useRef(false)

  useEffect(() => {
    if (!isResizing) return

    const onPointerMove = (e: PointerEvent) => {
      if (!draggingRef.current) return
      setWidth(Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, e.clientX)))
    }
    const onPointerUp = () => {
      draggingRef.current = false
      setIsResizing(false)
    }

    window.addEventListener("pointermove", onPointerMove)
    window.addEventListener("pointerup", onPointerUp)
    return () => {
      window.removeEventListener("pointermove", onPointerMove)
      window.removeEventListener("pointerup", onPointerUp)
    }
  }, [isResizing])

  const handleResizeStart = (e: React.PointerEvent) => {
    e.preventDefault()
    draggingRef.current = true
    setIsResizing(true)
  }

  return (
    <>
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-zinc-900/40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden
        />
      )}
      <aside
        style={sidebarOpen ? { width } : undefined}
        className={`fixed inset-y-0 left-0 z-40 w-72 overflow-hidden bg-white lg:static lg:z-0 lg:translate-x-0 ${
          isResizing ? "" : "transition-[width,transform] duration-200 ease-out"
        } ${sidebarOpen ? "translate-x-0 lg:w-72" : "-translate-x-full lg:w-0"}`}
      >
        <div className="relative flex h-full w-full flex-col border-r border-zinc-200">
          <div className="flex items-center justify-between gap-2 px-4 pt-4">
            <div className="flex items-center gap-2">
              <div className="flex size-8 items-center justify-center rounded-lg bg-gradient-to-br from-brand-500 to-brand-700 text-base">
                🤖
              </div>
              <span className="text-sm font-semibold text-zinc-800">공공조달 어시스턴트</span>
            </div>
            <button
              type="button"
              onClick={() => setSidebarOpen(false)}
              className="rounded-md p-1.5 text-zinc-400 hover:bg-zinc-100"
              aria-label="사이드바 닫기"
            >
              <CloseIcon className="size-4" />
            </button>
          </div>

          <div className="px-3 pt-4">
            <button
              type="button"
              onClick={() => {setSidebarOpen(false); navigate("/")}}
              className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-brand-200 bg-brand-50 px-3 py-2 text-sm font-medium text-brand-700 transition hover:bg-brand-100"
            >
              <PlusIcon className="size-4" />
              새로운 대화 시작
            </button>
          </div>

          <div className="mt-2 flex-1 overflow-y-auto px-2 pb-4">
            <p className="px-3 py-2 text-[11px] font-semibold tracking-wide text-zinc-400">대화 이력</p>
            <ConversationList />
          </div>

          <div
            onPointerDown={handleResizeStart}
            className="absolute inset-y-0 right-0 hidden w-1.5 cursor-col-resize touch-none hover:bg-brand-300/60 lg:block"
            role="separator"
            aria-orientation="vertical"
            aria-label="사이드바 크기 조절"
          />
        </div>
      </aside>
    </>
  )
}
