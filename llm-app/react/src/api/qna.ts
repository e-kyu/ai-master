import { API_BASE_URL, ApiError } from "./client"
import type { AgentProgressEvent, NoticeScanResult, QuestionRequest } from "../types"

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

// NoticeScanAgent: 백엔드 SSE 스트림을 읽으며 진행상태 콜백 호출, 최종 result 반환
export async function analyzeNoticeStreaming(
  request: QuestionRequest,
  onProgress: (event: AgentProgressEvent) => void,
): Promise<NoticeScanResult> {
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
  let buffer = ""
  let result: NoticeScanResult | null = null

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const lines = buffer.split("\n")
    buffer = lines.pop() ?? ""

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue
      try {
        const event: AgentProgressEvent = JSON.parse(line.slice(6))
        if (event.type === "result" && event.data) {
          result = event.data
        } else {
          onProgress(event)
        }
      } catch {
        // ignore malformed SSE lines
      }
    }
  }

  if (!result) throw new ApiError("분석 결과를 받지 못했습니다.", 0)
  return result
}
