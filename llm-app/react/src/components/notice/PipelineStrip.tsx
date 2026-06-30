import { useIsMobile } from "../../hooks/useIsMobile"

export type StepStatus = "waiting" | "running" | "done" | "failed"

export interface PipelineStep {
  label: string
  sub: string
  status: StepStatus
}

function StepNode({ step, compact }: { step: PipelineStep; compact?: boolean }) {
  const { status } = step
  const size = compact ? 26 : 34
  const iconSize = compact ? 12 : 16

  const icon = (() => {
    if (status === "done")
      return (
        <div style={{ width: size, height: size, borderRadius: "50%", background: "#E3F6EE", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <svg width={iconSize} height={iconSize} viewBox="0 0 24 24" fill="none" stroke="#0E9F6E" strokeWidth="2.4">
            <path d="M20 6 9 17l-5-5" />
          </svg>
        </div>
      )
    if (status === "running")
      return (
        <div style={{ width: size, height: size, borderRadius: "50%", background: "#EAF0FE", display: "flex", alignItems: "center", justifyContent: "center", animation: "ringPulse 1.8s ease-out infinite" }}>
          <div style={{ width: iconSize, height: iconSize, border: "2.4px solid #BFD0F6", borderTopColor: "#3457D5", borderRadius: "50%", animation: "spin 0.7s linear infinite" }} />
        </div>
      )
    if (status === "failed")
      return (
        <div style={{ width: size, height: size, borderRadius: "50%", background: "#FCF2E3", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <svg width={iconSize} height={iconSize} viewBox="0 0 24 24" fill="none" stroke="#E07B12" strokeWidth="2.4">
            <path d="M18 6 6 18M6 6l12 12" />
          </svg>
        </div>
      )
    return (
      <div style={{ width: size, height: size, borderRadius: "50%", background: "#F1F3F7", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div style={{ width: compact ? 8 : 12, height: compact ? 8 : 12, borderRadius: "50%", background: "#CDD2DB" }} />
      </div>
    )
  })()

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: compact ? 4 : 7, width: compact ? 72 : 110, flexShrink: 0 }}>
      {icon}
      <div style={{ textAlign: "center", lineHeight: 1.3 }}>
        <div style={{ fontSize: compact ? 10.5 : 12.5, fontWeight: status === "running" ? 700 : 600, color: status === "running" ? "#3457D5" : status === "failed" ? "#E07B12" : "#161A22", whiteSpace: "nowrap" }}>
          {step.label}
        </div>
        {!compact && (
          <div style={{ fontSize: 10.5, color: status === "running" ? "#3457D5" : "#8A93A6" }}>{step.sub}</div>
        )}
      </div>
    </div>
  )
}

function Connector({ from, to }: { from: StepStatus; to: StepStatus }) {
  if (from === "done" && to === "done")
    return <div style={{ flex: 1, height: 2, minWidth: 12, marginBottom: 24, background: "#D7F0E4" }} />
  if (from === "done" && to === "running")
    return (
      <div style={{
        flex: 1, height: 3, minWidth: 12, marginBottom: 24, borderRadius: 2,
        backgroundImage: "repeating-linear-gradient(90deg, #3457D5 0, #3457D5 6px, transparent 6px, transparent 11px)",
        backgroundSize: "22px 100%",
        animation: "pipelineFlow 0.7s linear infinite",
      }} />
    )
  return <div style={{ flex: 1, height: 2, minWidth: 12, marginBottom: 24, background: "#E6E9EF" }} />
}

interface Props {
  steps: PipelineStep[]
  summaryNode?: React.ReactNode
}

export function PipelineStrip({ steps, summaryNode }: Props) {
  const isMobile = useIsMobile()

  return (
    <section style={{
      flexShrink: 0,
      background: "#FFFFFF",
      borderBottom: "1px solid #E6E9EF",
      display: "flex",
      flexDirection: isMobile ? "column" : "row",
      alignItems: "stretch",
      gap: isMobile ? 10 : 18,
      padding: isMobile ? "10px 14px" : "14px 20px",
    }}>
      <div style={{ flex: 1, display: "flex", alignItems: "center", minWidth: 0, overflowX: "auto" }}>
        {steps.flatMap((step, i) => {
          const nodes: React.ReactNode[] = [<StepNode key={`s${i}`} step={step} compact={isMobile} />]
          if (i < steps.length - 1) nodes.push(<Connector key={`c${i}`} from={step.status} to={steps[i + 1].status} />)
          return nodes
        })}
      </div>
      {summaryNode && (
        <>
          {isMobile
            ? <div style={{ height: 1, background: "#E6E9EF", flexShrink: 0 }} />
            : <div style={{ width: 1, background: "#E6E9EF", flexShrink: 0 }} />
          }
          <div style={{ flexShrink: 0, display: "flex", alignItems: "center", paddingLeft: isMobile ? 0 : 4 }}>
            {summaryNode}
          </div>
        </>
      )}
    </section>
  )
}
