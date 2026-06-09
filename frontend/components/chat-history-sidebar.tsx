"use client"

import { Clock, MessageSquare, Plus, Trash2 } from "lucide-react"
import type { ChatSessionSummary } from "@/lib/types"

interface ChatHistorySidebarProps {
  sessions: ChatSessionSummary[]
  activeSessionId: string | null
  isLoading: boolean
  onNewChat: () => void
  onOpenSession: (sessionId: string) => void
  onDeleteSession: (sessionId: string) => void
}

function startOfLocalDay(date: Date) {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate())
}

function groupLabel(updatedAt: string) {
  const date = new Date(updatedAt)
  if (Number.isNaN(date.getTime())) return "更早"

  const today = startOfLocalDay(new Date())
  const target = startOfLocalDay(date)
  const diffDays = Math.floor(
    (today.getTime() - target.getTime()) / (24 * 60 * 60 * 1000)
  )

  if (diffDays <= 0) return "今天"
  if (diffDays === 1) return "昨天"
  if (diffDays < 7) return "7 天内"
  if (diffDays < 30) return "30 天内"
  return `${date.getFullYear()}-${date.getMonth() + 1}`
}

function groupedSessions(sessions: ChatSessionSummary[]) {
  const groups = new Map<string, ChatSessionSummary[]>()
  for (const session of sessions) {
    const label = groupLabel(session.updated_at)
    groups.set(label, [...(groups.get(label) || []), session])
  }
  return Array.from(groups.entries())
}

export function ChatHistorySidebar({
  sessions,
  activeSessionId,
  isLoading,
  onNewChat,
  onOpenSession,
  onDeleteSession,
}: ChatHistorySidebarProps) {
  return (
    <aside className="rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
      <button
        type="button"
        onClick={onNewChat}
        className="mb-4 inline-flex h-10 w-full items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white text-sm font-medium text-slate-800 shadow-sm transition-colors hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700"
      >
        <Plus className="h-4 w-4" />
        开启新对话
      </button>

      <div className="mb-3 flex items-center gap-2 px-1">
        <Clock className="h-4 w-4 text-blue-600" />
        <h2 className="text-sm font-semibold text-slate-900">历史聊天记录</h2>
      </div>

      {isLoading ? (
        <div className="space-y-2 px-1 text-xs text-slate-500">正在读取历史...</div>
      ) : sessions.length === 0 ? (
        <div className="rounded-lg border border-dashed border-slate-200 px-3 py-4 text-xs leading-5 text-slate-500">
          还没有历史聊天。发起第一次检索后，会自动保存到这里。
        </div>
      ) : (
        <div className="space-y-4">
          {groupedSessions(sessions).map(([label, group]) => (
            <section key={label} className="space-y-1">
              <h3 className="px-1 text-xs font-medium text-slate-400">{label}</h3>
              {group.map((session) => {
                const isActive = session.id === activeSessionId
                return (
                  <div
                    key={session.id}
                    className={`group flex items-start gap-1 rounded-lg border p-2 transition-colors ${
                      isActive
                        ? "border-blue-200 bg-blue-50"
                        : "border-transparent hover:border-slate-200 hover:bg-slate-50"
                    }`}
                  >
                    <button
                      type="button"
                      onClick={() => onOpenSession(session.id)}
                      className="min-w-0 flex-1 text-left"
                      aria-current={isActive ? "true" : undefined}
                    >
                      <div className="flex items-start gap-2">
                        <MessageSquare className="mt-0.5 h-3.5 w-3.5 flex-none text-slate-500" />
                        <p className="line-clamp-2 text-sm font-medium leading-5 text-slate-800">
                          {session.title}
                        </p>
                      </div>
                      <p className="mt-1 truncate pl-5 text-xs text-slate-500">
                        {session.last_message_preview || session.memory_preview || "空会话"}
                      </p>
                      <p className="mt-1 pl-5 text-[11px] text-slate-400">
                        {session.message_count} 条消息
                        {session.compressed_message_count > 0
                          ? ` · 已压缩 ${session.compressed_message_count} 条`
                          : ""}
                      </p>
                    </button>
                    <button
                      type="button"
                      onClick={() => onDeleteSession(session.id)}
                      className="inline-flex h-7 w-7 flex-none items-center justify-center rounded-md text-slate-300 opacity-0 transition-all hover:bg-red-50 hover:text-red-600 group-hover:opacity-100"
                      aria-label="删除历史聊天"
                      title="删除历史聊天"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                )
              })}
            </section>
          ))}
        </div>
      )}
    </aside>
  )
}
