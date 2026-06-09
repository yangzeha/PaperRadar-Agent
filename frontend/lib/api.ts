import {
  SearchRequest,
  SearchResponse,
  AgentStep,
  ProviderStatus,
  TranslationRequest,
  TranslationResponse,
  ChatSession,
  ChatSessionSummary,
} from "./types"

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

function asArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : []
}

function normalizeSearchResponse(value: unknown): SearchResponse {
  const raw = (value && typeof value === "object" ? value : {}) as Partial<SearchResponse>
  return {
    query: String(raw.query || ""),
    answer: String(raw.answer || ""),
    citations: asArray(raw.citations),
    papers: asArray(raw.papers),
    steps: asArray(raw.steps),
    rewrite_count: Number(raw.rewrite_count || 0),
    classification: raw.classification ?? null,
    session_id: raw.session_id ?? null,
    report_template_version: raw.report_template_version ?? null,
  }
}

function normalizeChatSessionSummary(value: unknown): ChatSessionSummary {
  const raw = (value && typeof value === "object" ? value : {}) as Partial<ChatSessionSummary>
  return {
    id: String(raw.id || ""),
    title: String(raw.title || "新聊天"),
    created_at: String(raw.created_at || new Date().toISOString()),
    updated_at: String(raw.updated_at || raw.created_at || new Date().toISOString()),
    message_count: Number(raw.message_count || 0),
    compressed_message_count: Number(raw.compressed_message_count || 0),
    last_message_preview: String(raw.last_message_preview || ""),
    memory_preview: String(raw.memory_preview || ""),
  }
}

function normalizeChatSession(value: unknown): ChatSession {
  const raw = (value && typeof value === "object" ? value : {}) as Partial<ChatSession>
  const memory = raw.memory || {
    summary: "",
    important_notes: [],
    compressed_message_count: 0,
    last_compressed_at: null,
  }
  return {
    id: String(raw.id || ""),
    title: String(raw.title || "新聊天"),
    created_at: String(raw.created_at || new Date().toISOString()),
    updated_at: String(raw.updated_at || raw.created_at || new Date().toISOString()),
    memory: {
      summary: String(memory.summary || ""),
      important_notes: asArray<string>(memory.important_notes),
      compressed_message_count: Number(memory.compressed_message_count || 0),
      last_compressed_at: memory.last_compressed_at ?? null,
    },
    messages: asArray(raw.messages).map((message, index) => {
      const rawMessage = (message && typeof message === "object" ? message : {}) as ChatSession["messages"][number]
      return {
        id: String(rawMessage.id || `message-${index}`),
        role: rawMessage.role === "assistant" ? "assistant" : "user",
        content: String(rawMessage.content || ""),
        created_at: String(rawMessage.created_at || new Date().toISOString()),
        response: rawMessage.response
          ? normalizeSearchResponse(rawMessage.response)
          : undefined,
      }
    }),
  }
}

export async function searchPapers(
  request: SearchRequest
): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE_URL}/api/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    let message = `Search failed (${response.status})`
    try {
      const errorJson = await response.json()
      message = errorJson.detail || message
    } catch {
      message = await response.text()
    }
    if (response.status === 429) {
      throw new Error("Rate limit reached. Please wait 30 seconds and try again.")
    }
    throw new Error(message)
  }

  return normalizeSearchResponse(await response.json())
}

export async function getProviderStatus(): Promise<ProviderStatus> {
  const response = await fetch(`${API_BASE_URL}/api/provider`)
  if (!response.ok) {
    throw new Error(`Provider status failed (${response.status})`)
  }
  return response.json()
}

export async function translateAnswer(
  request: TranslationRequest
): Promise<TranslationResponse> {
  const response = await fetch(`${API_BASE_URL}/api/translate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    let message = `Translation failed (${response.status})`
    try {
      const errorJson = await response.json()
      message = errorJson.detail || message
    } catch {
      message = await response.text()
    }
    throw new Error(message)
  }

  return response.json()
}

export async function getChatSessions(): Promise<ChatSessionSummary[]> {
  const response = await fetch(`${API_BASE_URL}/api/chat/sessions`)
  if (!response.ok) {
    throw new Error(`Chat sessions failed (${response.status})`)
  }
  const data = await response.json()
  return asArray(data.sessions).map(normalizeChatSessionSummary)
}

export async function createChatSession(title?: string): Promise<ChatSession> {
  const response = await fetch(`${API_BASE_URL}/api/chat/sessions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ title }),
  })
  if (!response.ok) {
    throw new Error(`Create chat session failed (${response.status})`)
  }
  return normalizeChatSession(await response.json())
}

export async function getChatSession(sessionId: string): Promise<ChatSession> {
  const response = await fetch(`${API_BASE_URL}/api/chat/sessions/${sessionId}`)
  if (!response.ok) {
    throw new Error(`Load chat session failed (${response.status})`)
  }
  return normalizeChatSession(await response.json())
}

export async function deleteChatSession(sessionId: string): Promise<{ deleted: boolean }> {
  const response = await fetch(`${API_BASE_URL}/api/chat/sessions/${sessionId}`, {
    method: "DELETE",
  })
  if (!response.ok) {
    throw new Error(`Delete chat session failed (${response.status})`)
  }
  return response.json()
}

export interface WebSocketCallbacks {
  onStep: (step: AgentStep) => void
  onResult: (response: SearchResponse) => void
  onError: (error: string) => void
}

export function createSearchWebSocket(
  callbacks: WebSocketCallbacks
): {
  send: (request: SearchRequest) => void
  close: () => void
} {
  const wsUrl = API_BASE_URL.replace(/^http/, "ws")
  const ws = new WebSocket(`${wsUrl}/ws/search`)

  ws.onopen = () => {
    console.log("WebSocket connected")
  }

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)

      if (data.type === "step") {
        callbacks.onStep(data.data as AgentStep)
      } else if (data.type === "result") {
        callbacks.onResult(data.data as SearchResponse)
      } else if (data.type === "error") {
        callbacks.onError(data.data?.message || data.data || "Unknown error")
      }
    } catch (err) {
      callbacks.onError(`Failed to parse message: ${err}`)
    }
  }

  ws.onerror = () => {
    callbacks.onError("WebSocket connection error")
  }

  ws.onclose = () => {
    console.log("WebSocket disconnected")
  }

  return {
    send: (request: SearchRequest) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(request))
      } else {
        callbacks.onError("WebSocket is not connected")
      }
    },
    close: () => {
      ws.close()
    },
  }
}
