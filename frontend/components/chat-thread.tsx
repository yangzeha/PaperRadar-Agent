"use client"

import { Bot, User } from "lucide-react"
import { AnswerPanel } from "@/components/answer-panel"
import { SourceList } from "@/components/source-list"
import { ThinkingSteps } from "@/components/thinking-steps"
import type { ChatSession } from "@/lib/types"

interface ChatThreadProps {
  session: ChatSession
}

function formatMessageTime(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ""
  return date.toLocaleString("zh-CN", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })
}

export function ChatThread({ session }: ChatThreadProps) {
  const messages = Array.isArray(session.messages) ? session.messages : []

  if (messages.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-slate-200 bg-white px-4 py-8 text-center text-sm text-slate-500">
        当前是一个新对话，发送第一条消息后会显示完整聊天记录。
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {messages.map((message, index) => {
        const timestamp = formatMessageTime(message.created_at)
        const messageKey = message.id || `${message.role}-${index}`

        if (message.role === "user") {
          return (
            <article key={messageKey} className="flex justify-end">
              <div className="max-w-[min(760px,92%)]">
                <div className="mb-1 flex items-center justify-end gap-2 text-xs text-slate-400">
                  {timestamp && <span>{timestamp}</span>}
                  <User className="h-3.5 w-3.5" />
                </div>
                <div className="rounded-2xl rounded-tr-md border border-blue-100 bg-blue-50 px-4 py-3 text-sm leading-6 text-slate-900 shadow-sm">
                  {String(message.content || "")}
                </div>
              </div>
            </article>
          )
        }

        const response = message.response
        return (
          <article key={messageKey} className="flex justify-start">
            <div className="w-full max-w-4xl">
              <div className="mb-2 flex items-center gap-2 text-xs text-slate-400">
                <Bot className="h-3.5 w-3.5" />
                <span>PaperRadar-Agent</span>
                {timestamp && <span>{timestamp}</span>}
              </div>
              <div className="space-y-4">
                <AnswerPanel
                  answer={String(response?.answer || message.content || "")}
                  citations={Array.isArray(response?.citations) ? response.citations : []}
                  title={response?.classification === "general" ? "助手回答" : "论文雷达报告"}
                  workflow={
                    Array.isArray(response?.steps) && response.steps.length > 0 ? (
                      <ThinkingSteps
                        steps={response.steps}
                        defaultOpen={false}
                        variant="embedded"
                      />
                    ) : null
                  }
                />
                {Array.isArray(response?.papers) && response.papers.length > 0 && (
                  <SourceList papers={response.papers} />
                )}
              </div>
            </div>
          </article>
        )
      })}
    </div>
  )
}
