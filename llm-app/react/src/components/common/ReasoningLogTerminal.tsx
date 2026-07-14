import { useEffect, useRef, useState } from "react"
import type { ReasoningLogEvent, ReasoningPhase } from "../../types"

const PHASE_META: Record<ReasoningPhase, { label: string; color: string; icon: string }> = {
  planning:   { label: "PLANNING",   color: "#22D3EE", icon: "🧠" },
  evaluating: { label: "EVALUATING", color: "#FBBF24", icon: "🧐" },
  llm_call:   { label: "LLM CALL",   color: "#60A5FA", icon: "🤖" },
  mcp_call:   { label: "MCP CALL",   color: "#A78BFA", icon: "🔌" },
  correction: { label: "CORRECTION", color: "#F87171", icon: "🔄" },
  execution:  { label: "EXECUTION",  color: "#34D399", icon: "🛠️" },
}

interface Props {
  logs: ReasoningLogEvent[]
  active?: boolean
  defaultOpen?: boolean
}

export function ReasoningLogTerminal({ logs, active, defaultOpen = true }: Props) {
  const [open, setOpen] = useState(defaultOpen)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (open) bottomRef.current?.scrollIntoView({ block: "nearest" })
  }, [logs, open])

  const latest = logs[logs.length - 1]
  const latestMeta = latest ? PHASE_META[latest.phase] : undefined

  return (
    <div style={{ borderRadius: 12, border: "1px solid #1F2937", background: "#0B0F17", overflow: "hidden" }}>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{
          width: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 10,
          padding: "9px 14px",
          background: "transparent",
          border: "none",
          cursor: "pointer",
          fontFamily: "inherit",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
          {active && (
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#34D399", animation: "dotPulse 1.6s ease-out infinite", display: "inline-block", flexShrink: 0 }} />
          )}
          <span style={{ fontSize: 12.5, fontWeight: 700, color: "#E5E9F0", fontFamily: "'IBM Plex Mono', monospace" }}>
            실행 로그 ({logs.length})
          </span>
          {latestMeta && (
            <span style={{ fontSize: 10.5, fontWeight: 700, color: latestMeta.color, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              {latestMeta.icon} {latestMeta.label}
            </span>
          )}
        </div>
        <svg
          width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#8A93A6" strokeWidth="2.4"
          style={{ flexShrink: 0, transform: open ? "rotate(180deg)" : "none", transition: "transform .15s" }}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {open && (
        <div style={{ maxHeight: 260, overflowY: "auto", padding: "0 14px 12px", borderTop: "1px solid #1F2937" }}>
          {logs.length === 0 ? (
            <div style={{ padding: "12px 0", fontSize: 12, color: "#5B6577", fontFamily: "'IBM Plex Mono', monospace" }}>
              에이전트 실행을 기다리는 중...
            </div>
          ) : (
            logs.map((log, i) => {
              const meta = PHASE_META[log.phase]
              const isError = log.level === "error"
              return (
                <pre
                  key={i}
                  style={{
                    margin: "10px 0 0",
                    padding: 0,
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-word",
                    fontFamily: "'IBM Plex Mono', monospace",
                    fontSize: 11.5,
                    lineHeight: 1.7,
                    color: isError ? "#F87171" : meta.color,
                    opacity: isError ? 1 : 0.92,
                  }}
                >
                  {log.message}
                </pre>
              )
            })
          )}
          <div ref={bottomRef} />
        </div>
      )}
    </div>
  )
}
