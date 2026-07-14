import { API_BASE_URL, ApiError } from "./client"
import type { AgentProgressEvent, AgentStreamEvent, NoticeScanResult, QuestionRequest, ReasoningLogEvent } from "../types"

// PpsAssistAgent: 백엔드가 SSE 스트림으로 응답 -> log 이벤트는 onLog, 답변 청크는 onChunk로 분기
export async function askQuestionStreaming(
  request: QuestionRequest,
  onChunk: (chunkText: string, fullTextSoFar: string) => void,
  onLog?: (event: ReasoningLogEvent) => void,
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
  let buffer = ""
  let fullText = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const lines = buffer.split("\n")
    buffer = lines.pop() ?? ""

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue
      try {
        const event: AgentStreamEvent = JSON.parse(line.slice(6))
        if (event.type === "log") {
          onLog?.(event)
        } else if (event.type === "answer_chunk") {
          fullText += event.content
          onChunk(event.content, fullText)
        }
        // "done" -> 별도 처리 없음 (스트림 종료로 자연 처리)
      } catch {
        // ignore malformed SSE lines
      }
    }
  }

  return fullText
}

// NoticeScanAgent: 백엔드 SSE 스트림을 읽으며 진행상태/추론로그 콜백 호출, 최종 result 반환
export async function analyzeNoticeStreaming(
  request: QuestionRequest,
  onProgress: (event: AgentProgressEvent) => void,
  onLog?: (event: ReasoningLogEvent) => void,
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
        const event: AgentStreamEvent = JSON.parse(line.slice(6))
        if (event.type === "result" && event.data) {
          result = event.data
        } else if (event.type === "log") {
          onLog?.(event)
        } else if (event.type === "progress") {
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
