export interface PaperResult {
  title: string
  authors: string[]
  abstract: string
  url: string
  source: "arxiv" | "pubmed" | "openalex" | "ieee" | "arxiv_openalex"
  published: string | null
  relevance_score: number | null
  route?: string | null
  role?: string | null
  role_label?: string | null
  priority?: number | null
  reason_tags?: string[]
  is_core?: boolean | null
  is_background?: boolean | null
  paper_identity?: string | null
}

export interface AgentStep {
  node: string
  status: "running" | "completed" | "skipped"
  detail: string
  duration_ms: number | null
}

export interface Citation {
  index: number
  title: string
  url: string
}

export interface SearchResponse {
  query: string
  answer: string
  citations: Citation[]
  papers: PaperResult[]
  steps: AgentStep[]
  rewrite_count: number
  classification?: string | null
  session_id?: string | null
  report_template_version?: string | null
}

export interface SearchRequest {
  query: string
  sources: string[]
  max_results: number
  session_id?: string | null
}

export interface TranslationRequest {
  text: string
  target_language: "zh" | "en"
}

export interface TranslationResponse {
  text: string
  target_language: "zh" | "en"
}

export interface ProviderStatus {
  provider: string
  mode: "mock" | "real"
  has_api_key: boolean
  label: string
  key_env: string
}

export interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  created_at: string
  response?: SearchResponse
}

export interface ChatSessionMemory {
  summary: string
  important_notes: string[]
  compressed_message_count: number
  last_compressed_at: string | null
}

export interface ChatSession {
  id: string
  title: string
  created_at: string
  updated_at: string
  messages: ChatMessage[]
  memory: ChatSessionMemory
}

export interface ChatSessionSummary {
  id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
  compressed_message_count: number
  last_message_preview: string
  memory_preview: string
}
