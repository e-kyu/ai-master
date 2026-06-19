import { createContext, useContext } from "react"
import type { Agent, ChatMessage } from "../types"

export interface AppStateContextValue {
  selectedAgent: Agent | null
  convrstnId: string
  messages: ChatMessage[]
  enableExtDocse: boolean
  fileFullPath: string
  sidebarOpen: boolean
  historyVersion: number
  startNewConversation: (agent: Agent) => void
  resumeConversation: (agent: Agent, convrstnId: string, messages: ChatMessage[]) => void
  appendMessage: (message: ChatMessage) => void
  updateLastMessageAnswer: (answer: string) => void
  setEnableExtDocse: (value: boolean) => void
  setFileFullPath: (path: string) => void
  setSidebarOpen: (open: boolean) => void
  refreshHistory: () => void
}

export const AppStateContext = createContext<AppStateContextValue | null>(null)

export function useAppState() {
  const ctx = useContext(AppStateContext)
  if (!ctx) throw new Error("useAppState must be used within AppStateProvider")
  return ctx
}
