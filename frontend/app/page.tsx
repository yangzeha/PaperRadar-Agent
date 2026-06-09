"use client"

import { useState, useCallback, useEffect, useRef } from "react"
import { motion } from "framer-motion"
import { Brain } from "lucide-react"
import { Header } from "@/components/header"
import { SearchBar } from "@/components/search-bar"
import { ChatHistorySidebar } from "@/components/chat-history-sidebar"
import { ChatThread } from "@/components/chat-thread"
import { Skeleton } from "@/components/ui/skeleton"
import { RotatingText } from "@/components/ui/rotating-text"
import {
  deleteChatSession,
  getChatSession,
  getChatSessions,
  searchPapers,
} from "@/lib/api"
import type {
  SearchRequest,
  SearchResponse,
  AgentStep,
  ChatSession,
  ChatSessionSummary,
} from "@/lib/types"

const ROTATING_WORDS = [
  "论文雷达",
  "选题追踪",
  "研究趋势",
  "阅读路线",
  "项目建议",
]

const ROTATING_COLORS = [
  "#60A5FA", // blue
  "#A78BFA", // purple
  "#34D399", // emerald
  "#F472B6", // pink
  "#FBBF24", // amber
]

const PAPERRADAR_UI_VERSION = "paper-radar-card-v3"

const DEFAULT_INTRO_ANSWER = `# PaperRadar-Agent 功能介绍

你好，我是 **PaperRadar-Agent**，一个面向研究生和 AI Agent 学习者的中文论文雷达与选题追踪助手。

我可以帮你做这些事：

## 1. 研究方向雷达
你可以输入一个研究方向，例如：
- 智能体式检索增强生成
- 大模型智能体记忆机制
- 多智能体科研协作
- RAG 幻觉与引用可靠性评估

我会自动从 arXiv / PubMed 检索相关论文，并生成中文方向分析报告。

## 2. 方法路线分类
我会把检索到的论文按研究路线归类，例如：
- Survey / Taxonomy
- Planning / Reasoning
- Multi-Agent / Hierarchical RAG
- Evaluation / Benchmark
- Multimodal RAG
- Domain-specific Agentic RAG

这样你不用一篇篇乱读，可以先知道这个方向有哪些主线。

## 3. 代表论文推荐
我会从检索结果中挑出适合优先阅读的论文，并说明：
- 为什么值得读
- 属于哪条研究路线
- 适合入门、综述、评测还是项目选题
- 读完能产出什么

## 4. 研究空白分析
我会基于当前论文摘要和分类结果，总结可能的研究 gap，例如：
- Agentic RAG 评测指标不统一
- 多 Agent 检索协作的成本和稳定性
- 长期记忆污染与遗忘机制
- Citation grounding 与幻觉检测
- 多模态 Agentic RAG 的证据对齐问题

## 5. 两周阅读路线
如果你想快速入门某个方向，我会按阶段生成两周计划：
- 第 1-2 天：建立方向全景
- 第 3-4 天：理解核心机制
- 第 5-7 天：阅读代表方法
- 第 8-10 天：关注评测与问题
- 第 11-14 天：形成小项目方案

## 6. 小项目建议
我会结合论文趋势，给出两周内可以完成的 Agent 项目，例如：
- LangGraph Agentic RAG 评测器
- Citation Grounding Checker
- 长期记忆污染检测 RAG Agent
- 多 Agent 文献研究助手

## 7. 长期记忆
我可以保存你的关注方向、收藏论文和历史检索主题。后续你继续提问时，我会结合你的长期关注方向给出更贴合的推荐。

你可以直接输入：
“帮我调研一下智能体式 RAG 方向”
或者：
“我想两周内入门大模型智能体记忆机制，推荐论文和学习路线”`

const DEFAULT_INTRO_RESPONSE: SearchResponse = {
  query: "功能介绍",
  answer: DEFAULT_INTRO_ANSWER,
  citations: [],
  papers: [],
  steps: [],
  rewrite_count: 0,
  classification: "general",
  report_template_version: PAPERRADAR_UI_VERSION,
}

const DEFAULT_INTRO_SESSION: ChatSession = {
  id: "default-intro-session",
  title: "功能介绍",
  created_at: "2026-06-09T00:00:00+08:00",
  updated_at: "2026-06-09T00:00:00+08:00",
  memory: {
    summary: "",
    important_notes: [],
    compressed_message_count: 0,
    last_compressed_at: null,
  },
  messages: [
    {
      id: "default-intro-user",
      role: "user",
      content: "功能介绍",
      created_at: "2026-06-09T00:00:00+08:00",
    },
    {
      id: "default-intro-assistant",
      role: "assistant",
      content: DEFAULT_INTRO_ANSWER,
      created_at: "2026-06-09T00:00:01+08:00",
      response: DEFAULT_INTRO_RESPONSE,
    },
  ],
}

const LEGACY_SESSION_MARKERS = [
  "mock/fallback 模式",
  "当前 mock",
  "当前 fallback",
  "论文数量一致性检查器",
  "年份过滤核查器",
  "摘要证据表",
  "先按标题和摘要识别与用户问题最接近的论文",
  "第 1-2 天：先读第 1-3 篇",
  "Agentic RAG 方向论文雷达",
  "LLM Agent 长期记忆",
  "LangGraph Agentic RAG",
  "Multi-Agent Collaboration",
  "RAG Hallucination Evaluation",
]

const PROJECT_STORAGE_KEY_PATTERNS = [
  /^paperradar/i,
  /^paper-radar/i,
  /^scholar-agent/i,
  /^chat-session/i,
]

function isLegacyPollutingSession(session: ChatSessionSummary | ChatSession): boolean {
  const messageText =
    "messages" in session && Array.isArray(session.messages)
      ? session.messages.map((message) => String(message.content || "")).join("\n")
      : ""
  const combined = [
    session.title || "",
    "last_message_preview" in session ? session.last_message_preview || "" : "",
    "memory_preview" in session ? session.memory_preview || "" : "",
    messageText,
  ].join("\n")
  return LEGACY_SESSION_MARKERS.some((marker) => combined.includes(marker))
}

function migrateBrowserStateIfNeeded() {
  if (typeof window === "undefined") return
  const versionKey = "paperradar-ui-version"
  const currentVersion = window.localStorage.getItem(versionKey)
  const containsLegacyState =
    storageContainsLegacyMarkers(window.localStorage) ||
    storageContainsLegacyMarkers(window.sessionStorage)
  if (currentVersion === PAPERRADAR_UI_VERSION && !containsLegacyState) return

  for (let index = window.localStorage.length - 1; index >= 0; index -= 1) {
    const key = window.localStorage.key(index)
    if (key && PROJECT_STORAGE_KEY_PATTERNS.some((pattern) => pattern.test(key))) {
      window.localStorage.removeItem(key)
    }
  }
  for (let index = window.sessionStorage.length - 1; index >= 0; index -= 1) {
    const key = window.sessionStorage.key(index)
    if (key && PROJECT_STORAGE_KEY_PATTERNS.some((pattern) => pattern.test(key))) {
      window.sessionStorage.removeItem(key)
    }
  }

  const indexedDBHandle = window.indexedDB as IDBFactory & {
    databases?: () => Promise<Array<{ name?: string }>>
  }
  if (indexedDBHandle && typeof indexedDBHandle.databases === "function") {
    indexedDBHandle
      .databases()
      .then((databases) => {
        databases
          .map((database) => database.name)
          .filter((name): name is string => Boolean(name))
          .filter((name) => PROJECT_STORAGE_KEY_PATTERNS.some((pattern) => pattern.test(name)))
          .forEach((name) => indexedDBHandle.deleteDatabase(name))
      })
      .catch(() => undefined)
  }

  window.localStorage.setItem(versionKey, PAPERRADAR_UI_VERSION)
}

function storageContainsLegacyMarkers(storage: Storage): boolean {
  for (let index = 0; index < storage.length; index += 1) {
    const key = storage.key(index)
    if (!key || !PROJECT_STORAGE_KEY_PATTERNS.some((pattern) => pattern.test(key))) {
      continue
    }
    const value = storage.getItem(key) || ""
    if (LEGACY_SESSION_MARKERS.some((marker) => key.includes(marker) || value.includes(marker))) {
      return true
    }
  }
  return false
}

function latestResponseFromSession(session: ChatSession): SearchResponse | null {
  const messages = Array.isArray(session.messages) ? session.messages : []
  const assistantMessage = [...messages]
    .reverse()
    .find((message) => message.role === "assistant" && message.response)
  return assistantMessage?.response || null
}

export default function Home() {
  const [isLoading, setIsLoading] = useState(false)
  const [response, setResponse] = useState<SearchResponse | null>(null)
  const [steps, setSteps] = useState<AgentStep[]>([])
  const [error, setError] = useState<string | null>(null)
  const [sessions, setSessions] = useState<ChatSessionSummary[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [activeSession, setActiveSession] = useState<ChatSession | null>(DEFAULT_INTRO_SESSION)
  const [isLoadingSessions, setIsLoadingSessions] = useState(false)
  const chatScrollRef = useRef<HTMLDivElement | null>(null)
  const pendingAutoScrollRef = useRef(false)

  const refreshSessions = useCallback(async () => {
    setIsLoadingSessions(true)
    try {
      const loadedSessions = await getChatSessions()
      setSessions(loadedSessions.filter((session) => !isLegacyPollutingSession(session)))
    } catch (err) {
      setError(err instanceof Error ? err.message : "历史聊天读取失败")
    } finally {
      setIsLoadingSessions(false)
    }
  }, [])

  useEffect(() => {
    migrateBrowserStateIfNeeded()
    refreshSessions()
  }, [refreshSessions])

  const scrollChatToBottom = useCallback((behavior: ScrollBehavior = "smooth") => {
    window.requestAnimationFrame(() => {
      const container = chatScrollRef.current
      if (!container) return
      container.scrollTo({
        top: 0,
        behavior,
      })
    })
  }, [])

  const scheduleScrollChatToBottom = useCallback(
    (behavior: ScrollBehavior = "smooth") => {
      pendingAutoScrollRef.current = true
      ;[0, 120, 420, 900, 1800, 3000].forEach((delay) => {
        window.setTimeout(() => {
          scrollChatToBottom(behavior)
        }, delay)
      })
      window.setTimeout(() => {
        pendingAutoScrollRef.current = false
      }, 3400)
    },
    [scrollChatToBottom]
  )

  const handleSearch = useCallback(async (request: SearchRequest) => {
    setIsLoading(true)
    setError(null)
    setResponse(null)
    setSteps([])

    const simulatedNodes = [
      { node: "Router", detail: `识别任务类型: "${request.query}"` },
      {
        node: "Retriever",
        detail: `检索 ${request.sources.join(", ")} 论文...`,
      },
      { node: "Grader", detail: "评估论文相关性..." },
      { node: "Generator", detail: "生成中文论文雷达报告..." },
      { node: "HallucinationChecker", detail: "检查回答是否有依据..." },
      { node: "Synthesizer", detail: "整理引用和最终输出..." },
    ]

    const stepTimers: NodeJS.Timeout[] = []
    simulatedNodes.forEach((nodeInfo, index) => {
      const timer = setTimeout(() => {
        setSteps((prev) => {
          const updated = prev.map((s) =>
            s.status === "running"
              ? {
                  ...s,
                  status: "completed" as const,
                  duration_ms: 200 + Math.random() * 800,
                }
              : s
          )
          return [
            ...updated,
            {
              node: nodeInfo.node,
              status: "running" as const,
              detail: nodeInfo.detail,
              duration_ms: null,
            },
          ]
        })
      }, index * 600)
      stepTimers.push(timer)
    })

    try {
      const result = await searchPapers({
        ...request,
        session_id: activeSessionId,
      })
      stepTimers.forEach(clearTimeout)

      let finalSteps: AgentStep[]
      if (result.steps && result.steps.length > 0) {
        finalSteps = result.steps
      } else {
        finalSteps = simulatedNodes.map((nodeInfo, i) => ({
          node: nodeInfo.node,
          status: "completed" as const,
          detail: nodeInfo.detail,
          duration_ms: 300 + i * 150,
        }))
      }

      setSteps(finalSteps)
      setResponse(result)
      if (result.session_id) {
        setActiveSessionId(result.session_id)
        const session = await getChatSession(result.session_id)
        setActiveSession(session)
        scheduleScrollChatToBottom("smooth")
      }
      await refreshSessions()
    } catch (err) {
      stepTimers.forEach(clearTimeout)
      const message =
        err instanceof Error ? err.message : "发生未知错误"
      setError(message)

      // Mark all running/simulated steps as completed so nothing stays stuck
      setSteps((prev) =>
        prev.map((s) =>
          s.status === "running"
            ? { ...s, status: "completed" as const, detail: "失败: " + message.slice(0, 60), duration_ms: 0 }
            : s
        )
      )
    } finally {
      setIsLoading(false)
    }
  }, [activeSessionId, refreshSessions, scheduleScrollChatToBottom])

  const handleNewChat = useCallback(async () => {
    setIsLoading(false)
    setError(null)
    setResponse(DEFAULT_INTRO_RESPONSE)
    setSteps([])
    setActiveSessionId(null)
    setActiveSession(DEFAULT_INTRO_SESSION)
    scheduleScrollChatToBottom("auto")
    await refreshSessions()
  }, [refreshSessions, scheduleScrollChatToBottom])

  const handleOpenSession = useCallback(async (sessionId: string) => {
    setIsLoading(false)
    setError(null)
    const session = await getChatSession(sessionId)
    if (isLegacyPollutingSession(session)) {
      setActiveSessionId(null)
      setActiveSession(DEFAULT_INTRO_SESSION)
      setResponse(DEFAULT_INTRO_RESPONSE)
      setSteps([])
      await refreshSessions()
      scheduleScrollChatToBottom("auto")
      return
    }
    const restoredResponse = latestResponseFromSession(session)
    setActiveSessionId(session.id)
    setActiveSession(session)
    setResponse(restoredResponse)
    setSteps(restoredResponse?.steps || [])
    scheduleScrollChatToBottom("auto")
  }, [refreshSessions, scheduleScrollChatToBottom])

  const handleDeleteSession = useCallback(async (sessionId: string) => {
    await deleteChatSession(sessionId)
    if (activeSessionId === sessionId) {
      setActiveSessionId(null)
      setActiveSession(DEFAULT_INTRO_SESSION)
      setResponse(DEFAULT_INTRO_RESPONSE)
      setSteps([])
    }
    await refreshSessions()
  }, [activeSessionId, refreshSessions])

  const activeMessages = Array.isArray(activeSession?.messages)
    ? activeSession.messages
    : []
  const hasConversation = activeMessages.length > 0
  const hasResults = isLoading || response || steps.length > 0 || hasConversation
  const activeMessageCount = activeMessages.length

  useEffect(() => {
    if (!hasConversation) return

    scheduleScrollChatToBottom("auto")
  }, [activeSessionId, activeMessageCount, hasConversation, scheduleScrollChatToBottom])

  useEffect(() => {
    if (!hasConversation) return

    const container = chatScrollRef.current
    if (!container || typeof ResizeObserver === "undefined") return

    const scrollIfPending = () => {
      if (pendingAutoScrollRef.current) {
        scrollChatToBottom("auto")
      }
    }

    const observer = new ResizeObserver(scrollIfPending)
    observer.observe(container)

    const content = container.querySelector("[data-testid='chat-content']")
    if (content) observer.observe(content)

    scrollIfPending()
    return () => observer.disconnect()
  }, [activeSessionId, activeMessageCount, hasConversation, scrollChatToBottom])

  useEffect(() => {
    if (!hasConversation || !isLoading) return
    scrollChatToBottom("smooth")
  }, [hasConversation, isLoading, scrollChatToBottom])

  return (
    <div className="relative flex h-screen flex-col overflow-hidden bg-white text-slate-950">
      <div className="fixed inset-0 z-0 bg-white" />

      <div className="relative z-10 flex h-full min-h-0 flex-col">
        <Header />

        <div className="container max-w-7xl flex-1 min-h-0 px-4 pb-3">
          <div className="grid h-full min-h-0 grid-cols-1 gap-6 lg:grid-cols-[280px_1fr]">
            <div className="min-h-0 pb-3 pt-4 lg:overflow-y-auto">
              <ChatHistorySidebar
                sessions={sessions}
                activeSessionId={activeSessionId}
                isLoading={isLoadingSessions}
                onNewChat={handleNewChat}
                onOpenSession={handleOpenSession}
                onDeleteSession={handleDeleteSession}
              />
            </div>

            <div className="flex min-h-0 min-w-0 flex-col">
              {hasResults && (
                <section
                  ref={chatScrollRef}
                  data-testid="chat-scroll-region"
                  className="flex min-h-0 flex-1 flex-col-reverse overflow-y-auto px-4 pb-5 pt-2"
                >
                  <div data-testid="chat-content" className="mx-auto max-w-5xl">
                    <main className="space-y-6 pb-2">
                      {isLoading && !hasConversation && !response ? (
                        <div className="space-y-4">
                          <Skeleton className="h-6 w-48 bg-slate-200" />
                          <Skeleton className="h-4 w-full bg-slate-200" />
                          <Skeleton className="h-4 w-full bg-slate-200" />
                          <Skeleton className="h-4 w-3/4 bg-slate-200" />
                          <Skeleton className="h-4 w-full bg-slate-200" />
                          <Skeleton className="h-4 w-5/6 bg-slate-200" />
                          <div className="pt-4">
                            <Skeleton className="mb-3 h-5 w-36 bg-slate-200" />
                            <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
                              <Skeleton className="h-40 w-full rounded-xl bg-slate-200" />
                              <Skeleton className="h-40 w-full rounded-xl bg-slate-200" />
                            </div>
                          </div>
                        </div>
                      ) : (
                        <>
                          {activeSession?.memory?.summary && (
                            <div className="rounded-xl border border-blue-100 bg-blue-50/60 p-4 text-sm text-slate-700">
                              <div className="mb-2 flex items-center gap-2 text-slate-900">
                                <Brain className="h-4 w-4 text-blue-600" />
                                <h3 className="font-semibold">压缩记忆</h3>
                                <span className="text-xs text-slate-500">
                                  已压缩 {activeSession.memory.compressed_message_count} 条较早消息
                                </span>
                              </div>
                              <p className="leading-6">{activeSession.memory.summary}</p>
                              {Array.isArray(activeSession.memory.important_notes) && activeSession.memory.important_notes.length > 0 && (
                                <div className="mt-3">
                                  <p className="mb-1 text-xs font-medium text-slate-500">
                                    重要记录
                                  </p>
                                  <ul className="space-y-1 pl-4 text-xs leading-5 text-slate-600">
                                    {activeSession.memory.important_notes.slice(-6).map((note) => (
                                      <li key={note} className="list-disc">
                                        {note}
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                            </div>
                          )}
                          {activeSession && hasConversation && (
                            <ChatThread session={activeSession} />
                          )}
                          {!hasConversation && response && (
                            <ChatThread
                              session={{
                                id: response.session_id || "current-response",
                                title: response.query,
                                created_at: new Date().toISOString(),
                                updated_at: new Date().toISOString(),
                                memory: {
                                  summary: "",
                                  important_notes: [],
                                  compressed_message_count: 0,
                                  last_compressed_at: null,
                                },
                                messages: [
                                  {
                                    id: "current-user",
                                    role: "user",
                                    content: response.query,
                                    created_at: new Date().toISOString(),
                                  },
                                  {
                                    id: "current-assistant",
                                    role: "assistant",
                                    content: response.answer,
                                    created_at: new Date().toISOString(),
                                    response,
                                  },
                                ],
                              }}
                            />
                          )}
                          {isLoading && hasConversation && (
                            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                              <div className="space-y-3">
                                <Skeleton className="h-4 w-40 bg-slate-200" />
                                <Skeleton className="h-4 w-full bg-slate-200" />
                                <Skeleton className="h-4 w-5/6 bg-slate-200" />
                              </div>
                            </div>
                          )}
                        </>
                      )}
                    </main>
                  </div>
                </section>
              )}

              <section
                data-testid="composer-region"
                className={
                  hasResults
                    ? "flex-none border-t border-slate-100 bg-white/95 px-4 pb-3 pt-3 backdrop-blur"
                    : "flex min-h-0 flex-1 items-center px-4 py-8"
                }
              >
                <div className="mx-auto w-full max-w-4xl">
                  {!hasResults && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.6, delay: 0.2 }}
                      className="mb-8 text-center"
                    >
                      <h2 className="mb-4 text-3xl font-bold text-slate-950 md:text-5xl lg:text-6xl">
                        PaperRadar-Agent{" "}
                        <RotatingText
                          texts={ROTATING_WORDS}
                          colors={ROTATING_COLORS}
                          interval={2500}
                        />
                      </h2>
                      <p className="mx-auto max-w-xl text-sm leading-relaxed text-slate-600 md:text-base">
                        面向 Agent 方向简历项目的中文论文雷达：检索 arXiv/PubMed，
                        用 LangGraph 完成路由、检索、评分、生成、幻觉检查和引用整理。
                      </p>
                    </motion.div>
                  )}

                  {error && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="mx-auto mb-3 max-w-3xl"
                    >
                      <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                        <strong>错误:</strong> {error}
                      </div>
                    </motion.div>
                  )}

                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.4 }}
                  >
                    <SearchBar onSearch={handleSearch} isLoading={isLoading} />
                  </motion.div>
                </div>
              </section>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
