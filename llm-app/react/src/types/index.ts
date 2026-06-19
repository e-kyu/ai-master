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
  postDate?: string
  agency?: string
  awardMethod?: string
  awardDetail?: string
  [key: string]: unknown
}

export interface NoticeExecutionInfo {
  bidEndDate?: string
  manager?: string
  openDate?: string
  openPlace?: string
  depositExemptYn?: string
  [key: string]: unknown
}

export interface NoticeItem {
  [key: string]: unknown
}

export interface NoticeExtractedData {
  general?: NoticeGeneralInfo
  execution?: NoticeExecutionInfo
  items?: NoticeItem[]
}

export interface NoticeScanResult {
  is_violating?: boolean
  extracted_data?: NoticeExtractedData
  [key: string]: unknown
}
