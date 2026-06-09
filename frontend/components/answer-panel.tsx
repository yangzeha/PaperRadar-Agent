"use client"

import { useEffect, useState } from "react"
import type { ReactNode } from "react"
import { motion } from "framer-motion"
import ReactMarkdown from "react-markdown"
import { FileText, Languages, Loader2 } from "lucide-react"
import { translateAnswer } from "@/lib/api"
import type { Citation } from "@/lib/types"

interface AnswerPanelProps {
  answer: string | null
  citations: Citation[]
  title?: string
  workflow?: ReactNode
}

function removeEmbeddedReferencesSection(answer: string): string {
  return answer.trim()
}

function splitCitationScopes(answer: string): string[] {
  const scopes: string[] = []
  let current: string[] = []

  const flushCurrent = () => {
    if (current.length === 0) return
    scopes.push(current.join("\n"))
    current = []
  }

  for (const line of answer.split("\n")) {
    const isBlank = line.trim() === ""
    const startsListItem = /^\s*(?:[-*+]\s+|\d+[.)、]\s+)/.test(line)
    const startsHeading = /^\s*#{1,6}\s+/.test(line)

    if ((isBlank || startsListItem || startsHeading) && current.length > 0) {
      flushCurrent()
    }

    if (isBlank) {
      scopes.push(line)
      continue
    }

    current.push(line)
  }

  flushCurrent()
  return scopes
}

function removeDuplicateCitations(answer: string): string {
  return splitCitationScopes(answer)
    .map((scope) => {
      const seen = new Set<string>()
      return scope.replace(/[ \t]*\[(\d+)\]/g, (match, num) => {
        if (seen.has(num)) return ""
        seen.add(num)
        return match
      })
    })
    .join("\n")
    .replace(/[ \t]+([,，.。;；:：!?！？])/g, "$1")
    .replace(/\n{3,}/g, "\n\n")
    .trim()
}

export function AnswerPanel({
  answer,
  citations,
  title = "论文雷达报告",
  workflow,
}: AnswerPanelProps) {
  const [language, setLanguage] = useState<"zh" | "en">("zh")
  const [translatedAnswer, setTranslatedAnswer] = useState<string | null>(null)
  const [isTranslating, setIsTranslating] = useState(false)
  const [translationError, setTranslationError] = useState<string | null>(null)

  useEffect(() => {
    setLanguage("zh")
    setTranslatedAnswer(null)
    setTranslationError(null)
  }, [answer])

  if (!answer) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-8 shadow-sm">
        <div className="flex flex-col items-center justify-center text-center">
          <FileText className="h-12 w-12 text-slate-300 mb-3" />
          <p className="text-sm text-slate-500">
            输入研究方向后，系统会生成带引用的中文论文雷达报告。
          </p>
        </div>
      </div>
    )
  }

  const answerWithoutEmbeddedReferences = removeDuplicateCitations(
    removeEmbeddedReferencesSection(answer)
  )
  const displayAnswer =
    language === "en" && translatedAnswer
      ? translatedAnswer
      : answerWithoutEmbeddedReferences

  const processedAnswer = displayAnswer.replace(
    /\[(\d+)\]/g,
    (match, num) => {
      const index = parseInt(num, 10)
      const citation = citations.find((c) => c.index === index)
      if (citation?.url) {
        return `[\\[${num}\\]](${citation.url})`
      }
      return match
    }
  )

  const handleToggleLanguage = async () => {
    if (language === "en") {
      setLanguage("zh")
      return
    }

    if (translatedAnswer) {
      setLanguage("en")
      return
    }

    setIsTranslating(true)
    setTranslationError(null)
    try {
      const result = await translateAnswer({
        text: answerWithoutEmbeddedReferences,
        target_language: "en",
      })
      setTranslatedAnswer(
        removeDuplicateCitations(removeEmbeddedReferencesSection(result.text))
      )
      setLanguage("en")
    } catch (error) {
      setTranslationError(
        error instanceof Error ? error.message : "翻译失败，请稍后重试"
      )
    } finally {
      setIsTranslating(false)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="rounded-2xl border border-slate-200 bg-white overflow-hidden shadow-sm"
    >
      <div className="px-5 py-4 border-b border-slate-100 flex items-center gap-2">
        <FileText className="h-4 w-4 text-blue-600" />
        <h3 className="text-sm font-semibold text-slate-800">{title}</h3>
        <button
          type="button"
          onClick={handleToggleLanguage}
          disabled={isTranslating}
          className="ml-auto inline-flex h-8 items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2.5 text-xs font-medium text-slate-600 transition-colors hover:border-blue-300 hover:text-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isTranslating ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Languages className="h-3.5 w-3.5" />
          )}
          {language === "zh" ? "中文报告" : "English"}
        </button>
      </div>

      {workflow && (
        <div className="border-b border-slate-100 px-5 py-4">
          {workflow}
        </div>
      )}

      <div className="px-5 py-4">
        {translationError && (
          <div className="mb-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
            {translationError}
          </div>
        )}
        <div className="prose-scholar text-sm leading-relaxed text-slate-700">
          <ReactMarkdown
            components={{
              a: ({ href, children, title }) => {
                const text = String(children)
                const isCitation = /^\[\d+\]$/.test(text)

                if (isCitation) {
                  return (
                    <a
                      href={href}
                      target="_blank"
                      rel="noopener noreferrer"
                      title={title || undefined}
                      className="inline-flex items-center justify-center min-w-[1.5rem] h-5 px-1 mx-0.5 text-xs font-semibold text-blue-700 bg-blue-50 rounded-md hover:bg-blue-100 transition-colors no-underline border border-blue-200"
                    >
                      {children}
                    </a>
                  )
                }

                return (
                  <a
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-700 hover:underline"
                  >
                    {children}
                  </a>
                )
              },
            }}
          >
            {processedAnswer}
          </ReactMarkdown>
        </div>

      </div>
    </motion.div>
  )
}
