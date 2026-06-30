export type AgentMode = "PpsAssistAgent" | "NoticeScanAgent" | string

export interface Agent {
  agent_id: string
  name: string
  description: string
  mode: AgentMode
  resource?: string | null
  created_at: string
}

export interface ConvrstnListItem {
  convrstn_id: string
  topic: string | null
  created_at: string
  agent_id: string | null
  mode: AgentMode | null
  name: string | null
  description: string | null
}

export interface ConvrstnDetail {
  convrstn_id: string
  created_at: string
  question: string
  answer: string | null
  answer_at: string | null
}

export interface ChatMessage {
  question: string
  answer: string
}

export interface QuestionRequest {
  agent_id: string
  agent_mode: AgentMode
  convrstnId: string
  fileFullPath: string
  question: string
  enableExtDocse: boolean
}

export interface UploadResponse {
  fileFullPath: string
  fileName: string
}

export interface NoticeGeneralInfo {
  noticeType?: string
  contractMethod?: string
  noticeName?: string
  noticeNo?: string
  refNo?: string
  postDate?: string
  agency?: string
  demandAgency?: string
  contractType?: string
  contractForm?: string
  bidMethod?: string
  stockType?: string
  awardMethod?: string
  awardDetail?: string
  rebidYn?: string
  [key: string]: unknown
}

export interface NoticeExecutionInfo {
  bidStartDate?: string
  bidEndDate?: string
  manager?: string
  openDate?: string
  openPlace?: string
  depositExemptYn?: string
  depositDate?: string
  relatedNotice?: string
  [key: string]: unknown
}

export interface NoticeItem {
  itemNo?: string
  itemName?: string
  standard?: string
  unit?: string
  quantity?: number | string | null
  unitPrice?: number | string | null
  amount?: number | string | null
  [key: string]: unknown
}

export interface NoticeExtractedData {
  general?: NoticeGeneralInfo
  execution?: NoticeExecutionInfo
  items?: NoticeItem[]
  progresses?: unknown[]
  statuses?: unknown[]
}

export interface NoticeScanResult {
  is_violating?: boolean
  extracted_data?: NoticeExtractedData
  document_text?: string
  status?: string
  error_message?: string | null
  [key: string]: unknown
}

export interface AgentProgressEvent {
  type: "progress" | "result"
  step?: string
  label?: string
  message?: string
  status?: "running" | "done" | "failed"
  data?: NoticeScanResult
}
