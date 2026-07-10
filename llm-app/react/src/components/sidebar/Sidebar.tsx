import { useEffect, useMemo, useRef, useState } from "react"
import { useNavigate } from "react-router-dom"
import { useAppState } from "../../context/appStateStore"
import { useConversationHistory } from "../../hooks/useConversationHistory"
import { CloseIcon, PlusIcon } from "../common/Icons"
import { ConversationList } from "./ConversationList"

const MIN_WIDTH = 240
const MAX_WIDTH = 480
const DEFAULT_WIDTH = 350

const HISTORY_GROUPS = [
  { mode: "NoticeScanAgent", label: "공고서 항목 자동 추출 Agent" },
  { mode: "PpsAssistAgent", label: "공공조달 전문 상담 Agent" },
] as const

export function Sidebar() {
  const navigate = useNavigate()
  const { sidebarOpen, setSidebarOpen, historyVersion } = useAppState()
  const [width, setWidth] = useState(DEFAULT_WIDTH)
  const [isResizing, setIsResizing] = useState(false)
  const draggingRef = useRef(false)
  const { items, loading, error, remove } = useConversationHistory(historyVersion)
  const [activeTab, setActiveTab] = useState<string>(HISTORY_GROUPS[0].mode)

  const tabs = useMemo(() => {
    const known = new Set<string>(HISTORY_GROUPS.map((g) => g.mode))
    const others = items.filter((item) => !item.mode || !known.has(item.mode))
    const groups = HISTORY_GROUPS.map((group) => ({
      id: group.mode,
      label: group.label,
      items: items.filter((item) => item.mode === group.mode),
    }))
    return others.length > 0 ? [...groups, { id: "others", label: "기타", items: others }] : groups
  }, [items])

  const activeGroup = tabs.find((tab) => tab.id === activeTab) ?? tabs[0]

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

          <div className="mt-3 flex gap-1 p-1 bg-zinc-100 rounded-lg overflow-x-auto no-scrollbar">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 min-w-0 rounded-md px-2 py-1.5 text-center text-xs font-medium transition-all duration-200 ${
                  activeTab === tab.id
                    ? "bg-white text-brand-600 shadow-sm"
                    : "text-zinc-500 hover:bg-zinc-200 hover:text-zinc-700"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <div className="mt-2 flex-1 overflow-y-auto px-2 pb-4">
            <ConversationList
              items={activeGroup?.items ?? []}
              loading={loading}
              error={error}
              emptyDescription={
                activeGroup && activeGroup.id !== "others" ? `${activeGroup.label}와의 대화를 시작해보세요.` : undefined
              }
              onDelete={remove}
            />
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
