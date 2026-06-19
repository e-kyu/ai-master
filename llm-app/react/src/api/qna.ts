import { API_BASE_URL, ApiError } from "./client"
import type { NoticeScanResult, QuestionRequest } from "../types"

// PpsAssistAgent: 백엔드가 text/plain 스트림으로 응답 -> 청크 단위로 onChunk 콜백 호출
export async function askQuestionStreaming(
  request: QuestionRequest,
  onChunk: (chunkText: string, fullTextSoFar: string) => void,
): Promise<string> {
  const res = await fetch(`${API_BASE_URL}/convrstn/question`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  })
  if (!res.ok || !res.body) {
    throw new ApiError(`POST /convrstn/question failed: ${res.status}`, res.status)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder("utf-8")
  let fullText = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    const chunkText = decoder.decode(value, { stream: true })
    if (chunkText) {
      fullText += chunkText
      onChunk(chunkText, fullText)
    }
  }

  return fullText
}

// NoticeScanAgent: 백엔드가 application/json 으로 한 번에 응답
export async function analyzeNotice(request: QuestionRequest): Promise<NoticeScanResult> {
  const res = await fetch(`${API_BASE_URL}/convrstn/question`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  })
  if (!res.ok) throw new ApiError(`POST /convrstn/question failed: ${res.status}`, res.status)
  return res.json() as Promise<NoticeScanResult>
}
