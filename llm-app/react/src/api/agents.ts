import { apiGet } from "./client"
import type { Agent } from "../types"

export function fetchAgentList(): Promise<Agent[]> {
  return apiGet<Agent[]>("/agents/")
}
