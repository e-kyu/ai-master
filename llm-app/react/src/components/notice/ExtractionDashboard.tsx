import { useMemo, useState } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { ItemsTable } from "./ItemsTable"
import type { NoticeScanResult } from "../../types"
import { useIsMobile } from "../../hooks/useIsMobile"

// ─── Field definitions ───────────────────────────────────────────────────────

const FIELD_DEFS: { key: string; label: string; section: "general" | "execution" }[] = [
  { key: "noticeNo",        label: "공고번호",      section: "general" },
  { key: "noticeName",      label: "공고명",        section: "general" },
  { key: "noticeType",      label: "공고 종류",     section: "general" },
  { key: "agency",          label: "공고기관",      section: "general" },
  { key: "demandAgency",    label: "수요기관",      section: "general" },
  { key: "contractType",    label: "계약 종류",     section: "general" },
  { key: "contractMethod",  label: "계약 방법",     section: "general" },
  { key: "bidMethod",       label: "입찰 방법",     section: "general" },
  { key: "awardMethod",     label: "낙찰 방법",     section: "general" },
  { key: "awardDetail",     label: "낙찰 상세",     section: "general" },
  { key: "stockType",       label: "비축 구분",     section: "general" },
  { key: "postDate",        label: "게시일시",      section: "general" },
  { key: "rebidYn",         label: "재입찰 여부",   section: "general" },
  { key: "manager",         label: "담당자",        section: "execution" },
  { key: "bidStartDate",    label: "입찰 시작",     section: "execution" },
  { key: "bidEndDate",      label: "입찰 마감",     section: "execution" },
  { key: "openDate",        label: "개찰일시",      section: "execution" },
  { key: "openPlace",       label: "개찰장소",      section: "execution" },
  { key: "depositExemptYn", label: "보증금 면제",   section: "execution" },
  { key: "depositDate",     label: "보증금 납부일", section: "execution" },
  { key: "relatedNotice",   label: "관련 공고",     section: "execution" },
]

interface FlatField {
  key: string
  label: string
  value: string | null
  status: "ok" | "review"
}

function flattenFields(result: NoticeScanResult): FlatField[] {
  const general = (result.extracted_data?.general ?? {}) as Record<string, unknown>
  const execution = (result.extracted_data?.execution ?? {}) as Record<string, unknown>
  return FIELD_DEFS.map(({ key, label, section }) => {
    const raw = section === "general" ? general[key] : execution[key]
    const value = raw !== null && raw !== undefined && raw !== "" ? String(raw) : null
    return { key, label, value, status: value ? "ok" : "review" }
  })
}

// ─── Sub-components ──────────────────────────────────────────────────────────

function OkBadge() {
  return (
    <span style={{ flexShrink: 0, display: "flex", alignItems: "center", gap: 4, fontSize: 11, fontWeight: 600, color: "#0B7C56", background: "#E9F7F0", border: "1px solid #C7E9D8", padding: "3px 9px", borderRadius: 20 }}>
      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#0E9F6E" strokeWidth="3"><path d="M20 6 9 17l-5-5" /></svg>
      추출 완료
    </span>
  )
}

function ReviewBadge() {
  return (
    <span style={{ flexShrink: 0, display: "flex", alignItems: "center", gap: 4, fontSize: 11, fontWeight: 600, color: "#C2410C", background: "#FCF2E3", border: "1px solid #F0D8AE", padding: "3px 9px", borderRadius: 20 }}>
      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#E07B12" strokeWidth="2.6"><path d="M12 9v4" /><path d="M12 17h.01" /><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" /></svg>
      검토 필요
    </span>
  )
}

function FieldCard({ field, selected, onSelect }: { field: FlatField; selected: boolean; onSelect: () => void }) {
  return (
    <div
      onClick={onSelect}
      style={{
        border: `1px solid ${selected ? "transparent" : "#E6E9EF"}`,
        boxShadow: selected ? "0 0 0 2px #3457D5, 0 4px 14px rgba(52,87,213,.14)" : "0 1px 2px rgba(20,26,34,.04)",
        borderRadius: 10,
        background: "#fff",
        padding: "12px 14px",
        cursor: "pointer",
        animation: "floatUp .22s ease both",
        transition: "box-shadow .15s, border-color .15s",
      }}
    >
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 10 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 3, minWidth: 0 }}>
          <span style={{ fontSize: 13, fontWeight: 700, letterSpacing: "-.2px", color: "#161A22" }}>{field.label}</span>
          <span style={{ fontSize: 10.5, color: "#9AA3B2", fontFamily: "'IBM Plex Mono', monospace" }}>{field.key}</span>
        </div>
        {field.status === "ok" ? <OkBadge /> : <ReviewBadge />}
      </div>

      {field.status === "ok" ? (
        <div style={{ marginTop: 9, fontSize: 14, fontWeight: 600, color: "#1A1F29", wordBreak: "break-all" }}>
          {field.value}
        </div>
      ) : (
        <div style={{ marginTop: 9, fontSize: 12.5, color: "#B0863A", fontStyle: "italic" }}>
          공고문에서 해당 항목을 확인할 수 없습니다.
        </div>
      )}
    </div>
  )
}

// ─── JSON syntax view ────────────────────────────────────────────────────────

type JsonVal = string | number | boolean | null | JsonVal[] | { [k: string]: JsonVal }

function JsonLine({ depth, k, v, comma }: { depth: number; k?: string; v: JsonVal; comma: boolean }) {
  const indent = "  ".repeat(depth)

  const renderVal = (val: JsonVal): React.ReactNode => {
    if (val === null) return <span style={{ color: "#C2410C", fontWeight: 600, background: "#FCF1E0", borderRadius: 3, padding: "0 4px" }}>null</span>
    if (typeof val === "boolean") return <span style={{ color: "#2952CC" }}>{String(val)}</span>
    if (typeof val === "number") return <span style={{ color: "#2952CC" }}>{val}</span>
    if (typeof val === "string") return <span style={{ color: "#0E7C53" }}>"{val}"</span>
    if (Array.isArray(val)) {
      if (val.length === 0) return <span style={{ color: "#5B6577" }}>[]</span>
      return null
    }
    return null
  }

  if (Array.isArray(v)) {
    if (v.length === 0) {
      return (
        <div style={{ display: "flex", whiteSpace: "pre", paddingLeft: 10 }}>
          {k && <><span style={{ color: "#7A4FC0" }}>"{k}"</span><span style={{ color: "#5B6577" }}>: </span></>}
          <span style={{ color: "#5B6577" }}>[]</span>
          {comma && <span style={{ color: "#5B6577" }}>,</span>}
        </div>
      )
    }
    return (
      <>
        <div style={{ display: "flex", whiteSpace: "pre", paddingLeft: 10 }}>
          {indent}{k && <><span style={{ color: "#7A4FC0" }}>"{k}"</span><span style={{ color: "#5B6577" }}>: </span></>}
          <span style={{ color: "#5B6577" }}>[</span>
        </div>
        {v.map((item, i) => (
          <JsonLine key={i} depth={depth + 1} v={item} comma={i < v.length - 1} />
        ))}
        <div style={{ display: "flex", whiteSpace: "pre", paddingLeft: 10 }}>
          {indent}<span style={{ color: "#5B6577" }}>]{comma ? "," : ""}</span>
        </div>
      </>
    )
  }

  if (v !== null && typeof v === "object") {
    const entries = Object.entries(v)
    return (
      <>
        <div style={{ display: "flex", whiteSpace: "pre", paddingLeft: 10 }}>
          {indent}{k && <><span style={{ color: "#7A4FC0" }}>"{k}"</span><span style={{ color: "#5B6577" }}>: </span></>}
          <span style={{ color: "#5B6577" }}>{"{"}</span>
        </div>
        {entries.map(([ek, ev], i) => (
          <JsonLine key={ek} depth={depth + 1} k={ek} v={ev as JsonVal} comma={i < entries.length - 1} />
        ))}
        <div style={{ display: "flex", whiteSpace: "pre", paddingLeft: 10 }}>
          {indent}<span style={{ color: "#5B6577" }}>{"}"}{ comma ? "," : ""}</span>
        </div>
      </>
    )
  }

  return (
    <div style={{ display: "flex", whiteSpace: "pre", paddingLeft: 10 }}>
      {indent}{k && <><span style={{ color: "#7A4FC0" }}>"{k}"</span><span style={{ color: "#5B6577" }}>: </span></>}
      {renderVal(v)}
      {comma && <span style={{ color: "#5B6577" }}>,</span>}
    </div>
  )
}

// ─── Donut summary ───────────────────────────────────────────────────────────

function DonutSummary({ ok, review, total }: { ok: number; review: number; total: number }) {
  const isMobile = useIsMobile()
  const size = isMobile ? 52 : 72
  const r = isMobile ? 20 : 30
  const sw = isMobile ? 6 : 8
  const C = 2 * Math.PI * r
  const pct = total > 0 ? ok / total : 0
  const dash = `${(pct * C).toFixed(1)} ${C.toFixed(1)}`
  const cx = size / 2
  const cy = size / 2

  return (
    <div style={{ display: "flex", alignItems: "center", gap: isMobile ? 10 : 16 }}>
      <div style={{ position: "relative", width: size, height: size, flexShrink: 0 }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          <circle cx={cx} cy={cy} r={r} fill="none" stroke="#EBEEF3" strokeWidth={sw} />
          <circle cx={cx} cy={cy} r={r} fill="none" stroke="#0E9F6E" strokeWidth={sw} strokeLinecap="round" strokeDasharray={dash} transform={`rotate(-90 ${cx} ${cy})`} />
        </svg>
        <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", lineHeight: 1 }}>
          <span style={{ fontSize: isMobile ? 12 : 17, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace" }}>{Math.round(pct * 100)}%</span>
          {!isMobile && <span style={{ fontSize: 9.5, color: "#8A93A6", marginTop: 2 }}>추출률</span>}
        </div>
      </div>
      <div style={{ display: "flex", gap: isMobile ? 5 : 8 }}>
        <div style={{ minWidth: isMobile ? 50 : 72, padding: isMobile ? "6px 10px" : "9px 13px", borderRadius: 9, background: "#F4F8FF", border: "1px solid #E2E9F8" }}>
          <div style={{ fontSize: isMobile ? 10 : 11, color: "#5B6577", marginBottom: 2 }}>{isMobile ? "전체" : "전체 항목"}</div>
          <div style={{ fontSize: isMobile ? 17 : 20, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace" }}>{total}</div>
        </div>
        <div style={{ minWidth: isMobile ? 50 : 72, padding: isMobile ? "6px 10px" : "9px 13px", borderRadius: 9, background: "#E9F7F0", border: "1px solid #C7E9D8" }}>
          <div style={{ fontSize: isMobile ? 10 : 11, color: "#0B7C56", marginBottom: 2 }}>{isMobile ? "완료" : "추출 완료"}</div>
          <div style={{ fontSize: isMobile ? 17 : 20, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace", color: "#0B7C56" }}>{ok}</div>
        </div>
        <div style={{ minWidth: isMobile ? 50 : 72, padding: isMobile ? "6px 10px" : "9px 13px", borderRadius: 9, background: "#FCF2E3", border: "1px solid #F0D8AE" }}>
          <div style={{ fontSize: isMobile ? 10 : 11, color: "#B5701A", marginBottom: 2 }}>{isMobile ? "검토" : "검토 필요"}</div>
          <div style={{ fontSize: isMobile ? 17 : 20, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace", color: "#C2410C" }}>{review}</div>
        </div>
      </div>
    </div>
  )
}

// ─── Main export ─────────────────────────────────────────────────────────────

interface Props {
  result: NoticeScanResult
  fileName: string
}

type FilterMode = "all" | "ok" | "review"
type ViewMode = "fields" | "json"
type PanelTab = "doc" | "result"

export function ExtractionDashboard({ result, fileName }: Props) {
  const isMobile = useIsMobile()
  const [viewMode, setViewMode] = useState<ViewMode>("fields")
  const [filter, setFilter] = useState<FilterMode>("all")
  const [selectedField, setSelectedField] = useState<string | null>(null)
  const [panelTab, setPanelTab] = useState<PanelTab>("result")
  const [copied, setCopied] = useState(false)

  const handleCopyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(result.extracted_data, null, 2)).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }

  const fields = useMemo(() => flattenFields(result), [result])
  const okCount = fields.filter((f) => f.status === "ok").length
  const reviewCount = fields.filter((f) => f.status === "review").length
  const total = fields.length

  const shownFields =
    filter === "ok" ? fields.filter((f) => f.status === "ok") :
    filter === "review" ? fields.filter((f) => f.status === "review") :
    fields

  const items = result.extracted_data?.items ?? []
  const docText = result.document_text ?? ""

  const seg = (active: boolean) => ({
    fontFamily: "inherit",
    fontSize: 12,
    fontWeight: 600,
    border: "none",
    borderRadius: 6,
    padding: "5px 14px",
    cursor: "pointer",
    transition: "all .12s",
    ...(active
      ? { background: "#fff", color: "#161A22", boxShadow: "0 1px 2px rgba(20,26,34,.1)" }
      : { background: "transparent", color: "#8A93A6" }),
  } as React.CSSProperties)

  const chip = (active: boolean, onBg: string, onBd: string, onColor: string) => ({
    fontFamily: "inherit",
    fontSize: 11.5,
    fontWeight: 600,
    borderRadius: 7,
    padding: "5px 11px",
    cursor: "pointer",
    transition: "all .12s",
    ...(active
      ? { background: onBg, border: `1px solid ${onBd}`, color: onColor }
      : { background: "#fff", border: "1px solid #E6E9EF", color: "#5B6577" }),
  } as React.CSSProperties)

  const panelTabStyle = (active: boolean) => ({
    flex: 1,
    padding: "11px 0",
    fontSize: 13,
    fontWeight: active ? 700 : 500,
    color: active ? "#3457D5" : "#8A93A6",
    background: "none",
    border: "none",
    borderBottom: active ? "2px solid #3457D5" : "2px solid transparent",
    cursor: "pointer",
    fontFamily: "inherit",
    transition: "color .15s",
  } as React.CSSProperties)

  const showDoc = !isMobile || panelTab === "doc"
  const showResult = !isMobile || panelTab === "result"

  return (
    <div style={{ display: "flex", flexDirection: isMobile ? "column" : "row", height: "100%", minHeight: 0, gap: isMobile ? 0 : 12, padding: isMobile ? 0 : 12, background: "#F4F6F9" }}>

      {/* Mobile panel tab switcher */}
      {isMobile && (
        <div style={{ flexShrink: 0, display: "flex", background: "#fff", borderBottom: "1px solid #E6E9EF", padding: "0 14px" }}>
          <button onClick={() => setPanelTab("doc")} style={panelTabStyle(panelTab === "doc")}>원본 문서</button>
          <button onClick={() => setPanelTab("result")} style={panelTabStyle(panelTab === "result")}>추출 결과</button>
        </div>
      )}

      {/* ── LEFT: original document ── */}
      {showDoc && (
        <section style={{ flex: "1 1 0", minWidth: 0, minHeight: 0, display: "flex", flexDirection: "column", background: "#fff", border: isMobile ? "none" : "1px solid #E6E9EF", borderRadius: isMobile ? 0 : 12, overflow: "hidden" }}>
          <div style={{ flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "space-between", padding: isMobile ? "10px 14px" : "12px 16px", borderBottom: "1px solid #EEF0F4" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 9, minWidth: 0 }}>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#5B6577" strokeWidth="2" style={{ flexShrink: 0 }}><path d="M14 3v4a1 1 0 0 0 1 1h4" /><path d="M5 21V5a2 2 0 0 1 2-2h8l5 5v13a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2z" /></svg>
              <span style={{ fontSize: 13.5, fontWeight: 700 }}>원본 비축공고서</span>
              {!isMobile && <span style={{ fontSize: 11, color: "#9AA3B2", fontFamily: "'IBM Plex Mono', monospace", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{fileName}</span>}
            </div>
            {!isMobile && (
              <div style={{ display: "flex", alignItems: "center", gap: 13, fontSize: 11, color: "#5B6577", flexShrink: 0 }}>
                <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
                  <span style={{ width: 11, height: 11, borderRadius: 3, background: "#EAF0FE", boxShadow: "inset 0 -2px 0 #A9C0F2", display: "inline-block" }} />
                  추출됨
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
                  <span style={{ width: 11, height: 11, borderRadius: 3, background: "#FCF1E0", boxShadow: "inset 0 -2px 0 #EBB765", display: "inline-block" }} />
                  검토 필요
                </span>
              </div>
            )}
          </div>
          <div style={{ flex: 1, minHeight: 0, overflowY: "auto", padding: isMobile ? "14px 14px" : "20px 28px", background: "#EEF1F5" }}>
            {docText ? (
              <div style={{ background: "#fff", borderRadius: 8, padding: isMobile ? "16px 16px" : "28px 32px", boxShadow: "0 1px 4px rgba(0,0,0,.06)", fontSize: 14, lineHeight: 1.8, color: "#1A1F29", minHeight: 200 }}>
                <div className="prose prose-sm max-w-none">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{docText}</ReactMarkdown>
                </div>
              </div>
            ) : (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#9AA3B2", fontSize: 13 }}>
                원문 텍스트가 없습니다.
              </div>
            )}
          </div>
        </section>
      )}

      {/* ── RIGHT: extraction result ── */}
      {showResult && (
        <section style={{ flex: "1 1 0", minWidth: 0, minHeight: 0, display: "flex", flexDirection: "column", background: "#fff", border: isMobile ? "none" : "1px solid #E6E9EF", borderRadius: isMobile ? 0 : 12, overflow: "hidden" }}>

          {/* Panel header */}
          <div style={{ flexShrink: 0, padding: isMobile ? "10px 14px" : "12px 16px", borderBottom: "1px solid #EEF0F4" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: viewMode === "fields" ? 11 : 0 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 9, minWidth: 0 }}>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#3457D5" strokeWidth="2" style={{ flexShrink: 0 }}><path d="m18 16 4-4-4-4" /><path d="m6 8-4 4 4 4" /><path d="m14.5 4-5 16" /></svg>
                <span style={{ fontSize: isMobile ? 13 : 13.5, fontWeight: 700 }}>{isMobile ? "추출 결과" : "입력항목 추출 결과"}</span>
              </div>
              <div style={{ display: "flex", padding: 3, background: "#F1F3F7", borderRadius: 8, gap: 2, flexShrink: 0 }}>
                <button onClick={() => setViewMode("fields")} style={seg(viewMode === "fields")}>필드</button>
                <button onClick={() => setViewMode("json")} style={seg(viewMode === "json")}>JSON</button>
              </div>
            </div>

            {viewMode === "fields" && (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  <button onClick={() => setFilter("all")} style={chip(filter === "all", "#161A22", "#161A22", "#fff")}>전체 {total}</button>
                  <button onClick={() => setFilter("ok")} style={chip(filter === "ok", "#E9F7F0", "#C7E9D8", "#0B7C56")}>완료 {okCount}</button>
                  <button onClick={() => setFilter("review")} style={chip(filter === "review", "#FCF2E3", "#F0D8AE", "#C2410C")}>{isMobile ? "검토" : "처리 필요"} {reviewCount}</button>
                </div>
                <span style={{ fontSize: 11, color: "#8A93A6", flexShrink: 0 }}>
                  추출률 <b style={{ color: "#0B7C56", fontFamily: "'IBM Plex Mono', monospace" }}>{total > 0 ? Math.round((okCount / total) * 100) : 0}%</b>
                </span>
              </div>
            )}
            
            {viewMode === "json" && (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  <button onClick={handleCopyJson} style={chip(copied, "#161A22", "#161A22", "#fff")}>{copied ? "복사됨" : "클립보드 복사"}</button>
                </div>
              </div>
            )}
          </div>

          {/* Field cards */}
          {viewMode === "fields" && (
            <div style={{ flex: 1, minHeight: 0, overflowY: "auto", padding: isMobile ? "10px 12px" : "12px 14px", display: "flex", flexDirection: "column", gap: 9, background: "#FAFBFC" }}>
              {shownFields.map((f) => (
                <FieldCard
                  key={f.key}
                  field={f}
                  selected={selectedField === f.key}
                  onSelect={() => setSelectedField((prev) => (prev === f.key ? null : f.key))}
                />
              ))}

              {/* Items section */}
              {filter === "all" && items.length > 0 && (
                <div style={{ marginTop: 8, border: "1px solid #E6E9EF", borderRadius: 10, background: "#fff", padding: "12px 14px", animation: "floatUp .22s ease both" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 10 }}>
                    <span style={{ fontSize: 13, fontWeight: 700, color: "#161A22" }}>품목 목록</span>
                    <span style={{ fontSize: 11, background: "#EAF0FE", color: "#3457D5", border: "1px solid #C8D6F7", borderRadius: 20, padding: "2px 8px", fontWeight: 600 }}>{items.length}개</span>
                  </div>
                  <ItemsTable items={items} />
                </div>
              )}

              {shownFields.length === 0 && (
                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 120, color: "#9AA3B2", fontSize: 13 }}>
                  해당하는 항목이 없습니다.
                </div>
              )}
            </div>
          )}

          {/* JSON view */}
          {viewMode === "json" && (
            <div style={{ flex: 1, minHeight: 0, overflowY: "auto", padding: isMobile ? "12px 14px" : "16px 18px", background: "#FBFCFD", fontFamily: "'IBM Plex Mono', monospace", fontSize: isMobile ? 11.5 : 12.5, lineHeight: 1.85 }}>
              <JsonLine depth={0} v={result.extracted_data as JsonVal} comma={false} />
            </div>
          )}
        </section>
      )}
    </div>
  )
}

export { DonutSummary }
