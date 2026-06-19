import { apiDelete, apiGet } from "./client"
import type { ConvrstnDetail, ConvrstnListItem } from "../types"

export function fetchConversationHistory(): Promise<ConvrstnListItem[]> {
  return apiGet<ConvrstnListItem[]>("/convrstnHistory/")
}

export function fetchConversationDetail(convrstnId: string): Promise<ConvrstnDetail[]> {
  return apiGet<ConvrstnDetail[]>(`/convrstnHistory/${convrstnId}`)
}

export function deleteConversation(convrstnId: string): Promise<void> {
  return apiDelete(`/convrstnHistory/${convrstnId}`)
}
