import { useCallback, useMemo, useState, type ReactNode } from "react"
import type { Agent, ChatMessage } from "../types"
import { AppStateContext, type AppStateContextValue } from "./appStateStore"

function newConvrstnId() {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null)
  const [convrstnId, setConvrstnId] = useState<string>(newConvrstnId())
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [enableExtDocse, setEnableExtDocse] = useState(false)
  const [fileFullPath, setFileFullPath] = useState("")
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [historyVersion, setHistoryVersion] = useState(0)

  const startNewConversation = useCallback((agent: Agent) => {
    setSelectedAgent(agent)
    setConvrstnId(newConvrstnId())
    setMessages([])
    setFileFullPath("")
    setSidebarOpen(false)
  }, [])

  const resumeConversation = useCallback(
    (agent: Agent, id: string, msgs: ChatMessage[]) => {
      setSelectedAgent(agent)
      setConvrstnId(id)
      setMessages(msgs)
      setFileFullPath("")
      setSidebarOpen(false)
    },
    [],
  )

  const appendMessage = useCallback((message: ChatMessage) => {
    setMessages((prev) => [...prev, message])
  }, [])

  const updateLastMessageAnswer = useCallback((answer: string) => {
    setMessages((prev) => {
      if (prev.length === 0) return prev
      const next = prev.slice()
      next[next.length - 1] = { ...next[next.length - 1], answer }
      return next
    })
  }, [])

  const refreshHistory = useCallback(() => {
    setHistoryVersion((v) => v + 1)
  }, [])

  const value = useMemo<AppStateContextValue>(
    () => ({
      selectedAgent,
      convrstnId,
      messages,
      enableExtDocse,
      fileFullPath,
      sidebarOpen,
      historyVersion,
      startNewConversation,
      resumeConversation,
      appendMessage,
      updateLastMessageAnswer,
      setEnableExtDocse,
      setFileFullPath,
      setSidebarOpen,
      refreshHistory,
    }),
    [
      selectedAgent,
      convrstnId,
      messages,
      enableExtDocse,
      fileFullPath,
      sidebarOpen,
      historyVersion,
      startNewConversation,
      resumeConversation,
      appendMessage,
      updateLastMessageAnswer,
      refreshHistory,
    ],
  )

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>
}
