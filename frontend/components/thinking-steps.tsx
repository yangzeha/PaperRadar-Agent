"use client"

import { useState, type ElementType } from "react"
import {
  Activity,
  Brain,
  CheckCircle2,
  ChevronUp,
  FileCheck,
  Loader2,
  PenLine,
  RefreshCw,
  Search,
  ShieldCheck,
  SkipForward,
  Sparkles,
} from "lucide-react"
import { cn } from "@/lib/utils"
import type { AgentStep } from "@/lib/types"

interface ThinkingStepsProps {
  steps: AgentStep[]
  defaultOpen?: boolean
  variant?: "card" | "embedded"
}

const NODE_CONFIG: Record<string, { label: string; icon: ElementType }> = {
  Router: { label: "任务路由", icon: Brain },
  router: { label: "任务路由", icon: Brain },
  Retriever: { label: "论文检索", icon: Search },
  retriever: { label: "论文检索", icon: Search },
  Grader: { label: "相关性评分", icon: FileCheck },
  grader: { label: "相关性评分", icon: FileCheck },
  Rewriter: { label: "查询改写", icon: RefreshCw },
  rewriter: { label: "查询改写", icon: RefreshCw },
  Generator: { label: "报告生成", icon: PenLine },
  generator: { label: "报告生成", icon: PenLine },
  HallucinationChecker: { label: "依据检查", icon: ShieldCheck },
  hallucination_checker: { label: "依据检查", icon: ShieldCheck },
  Synthesizer: { label: "引用整理", icon: Sparkles },
  synthesizer: { label: "引用整理", icon: Sparkles },
}

const CLASSIFICATION_LABELS: Record<string, string> = {
  paper_search: "论文检索",
  paper_qa: "论文问答",
  paper_radar: "论文雷达",
  reading_plan: "阅读路线",
  project_idea: "项目建议",
  general: "普通对话",
}

function getNodeConfig(nodeName: string) {
  return (
    NODE_CONFIG[nodeName] || {
      label: nodeName,
      icon: Brain,
    }
  )
}

function formatDuration(ms: number | null): string {
  if (ms === null || ms === 0) return ""
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function translateClassification(value: string): string {
  return CLASSIFICATION_LABELS[value] || value
}

function translateDetail(detail = ""): string {
  const text = detail.trim()
  if (!text) return "暂无详细信息"

  const classified = text.match(/^Classified as '([^']+)'$/i)
  if (classified) {
    return `已分类为“${translateClassification(classified[1])}”`
  }

  const mock = text.match(/^Mock retrieval returned (\d+) papers for:\s*(.*)$/i)
  if (mock) {
    return `模拟检索返回 ${mock[1]} 篇论文，查询：${mock[2]}`
  }

  const fetched = text.match(/^Fetched (\d+) papers, retrieved top (\d+)$/i)
  if (fetched) {
    return `已抓取 ${fetched[1]} 篇论文，向量检索保留前 ${fetched[2]} 篇`
  }

  const fetchedWithFilters = text.match(
    /^Fetched (\d+) papers, kept (\d+) after filters, retrieved top (\d+)$/i
  )
  if (fetchedWithFilters) {
    return `已抓取 ${fetchedWithFilters[1]} 篇论文，过滤后保留 ${fetchedWithFilters[2]} 篇，最终取前 ${fetchedWithFilters[3]} 篇`
  }

  if (/^No papers found from external APIs$/i.test(text)) {
    return "外部论文接口没有找到结果"
  }

  const yearFiltered = text.match(
    /^No papers found within (\d{4})-(\d{4}) after filtering external results$/i
  )
  if (yearFiltered) {
    return `外部接口返回的结果中没有 ${yearFiltered[1]}-${yearFiltered[2]} 年范围内的论文`
  }

  const relevant = text.match(/^(\d+)\/(\d+) documents relevant$/i)
  if (relevant) {
    return `共有 ${relevant[1]}/${relevant[2]} 篇文档被判断为相关`
  }

  const rewrite = text.match(/^Rewrite #(\d+): '(.+)'$/i)
  if (rewrite) {
    return `第 ${rewrite[1]} 次改写查询：${rewrite[2]}`
  }

  const generated = text.match(
    /^Generated (PaperRadar report|answer) with (\d+) citations$/i
  )
  if (generated) {
    const mode =
      generated[1].toLowerCase() === "paperradar report"
        ? "PaperRadar 报告"
        : "回答"
    return `已生成${mode}，包含 ${generated[2]} 条引用`
  }

  const hallucination = text.match(/^Hallucination score:\s*([0-9.]+)$/i)
  if (hallucination) {
    return `幻觉分数：${hallucination[1]}`
  }

  if (/^No source documents to check against$/i.test(text)) {
    return "没有可用于依据检查的来源文档"
  }

  const finalAnswer = text.match(
    /^Final answer:\s*(\d+)\s*chars,\s*(\d+)\s*citations$/i
  )
  if (finalAnswer) {
    return `最终答案：${finalAnswer[1]} 个字符，${finalAnswer[2]} 条引用`
  }

  return text
}

export function ThinkingSteps({
  steps,
  defaultOpen = true,
  variant = "card",
}: ThinkingStepsProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen)
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null)
  const safeSteps = Array.isArray(steps) ? steps : []

  if (safeSteps.length === 0) return null

  const completedCount = safeSteps.filter(
    (s) => s.status === "completed" || s.status === "skipped"
  ).length
  const isRunning = safeSteps.some((s) => s.status === "running")

  return (
    <div
      className={cn(
        "w-full overflow-hidden",
        variant === "card"
          ? "rounded-2xl border border-slate-200 bg-white shadow-sm"
          : "rounded-xl border border-slate-100 bg-slate-50/70",
        "transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)]",
        variant === "card" && (isOpen ? "rounded-3xl" : "rounded-2xl")
      )}
    >
      <button
        type="button"
        className="flex w-full items-center gap-3 p-4 text-left"
        onClick={() => setIsOpen((open) => !open)}
      >
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-100 transition-colors duration-300">
          {isRunning ? (
            <Loader2 className="h-4 w-4 animate-spin text-slate-600" />
          ) : (
            <Activity className="h-4 w-4 text-slate-600" />
          )}
        </div>
        <div className="flex-1 overflow-hidden">
          <h3 className="text-sm font-semibold text-slate-900">
            {isRunning ? "智能体流程运行中..." : `${completedCount} 个步骤已完成`}
          </h3>
          <p
            className={cn(
              "text-xs text-slate-500",
              "transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)]",
              isOpen ? "mt-0 max-h-0 opacity-0" : "mt-0.5 max-h-6 opacity-100"
            )}
          >
            {isRunning ? "正在处理你的研究方向" : "点击查看流程细节"}
          </p>
        </div>
        <div className="flex h-8 w-8 items-center justify-center">
          <ChevronUp
            className={cn(
              "h-4 w-4 text-slate-400 transition-transform duration-500 ease-[cubic-bezier(0.4,0,0.2,1)]",
              isOpen ? "rotate-0" : "rotate-180"
            )}
          />
        </div>
      </button>

      <div
        className={cn(
          "grid",
          "transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)]",
          isOpen ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
        )}
      >
        <div className="overflow-hidden">
          <div className="px-2 pb-3">
            <div className="space-y-0.5">
              {safeSteps.map((step, index) => {
                const config = getNodeConfig(String(step.node || ""))
                const NodeIcon = config.icon
                const isStepRunning = step.status === "running"
                const isCompleted = step.status === "completed"
                const isSkipped = step.status === "skipped"
                const isExpanded = expandedIndex === index
                const detail = translateDetail(step.detail)

                return (
                  <button
                    type="button"
                    key={`${step.node}-${index}`}
                    className={cn(
                      "flex w-full items-start gap-3 rounded-xl p-3 text-left",
                      "transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)]",
                      isStepRunning && "bg-blue-50",
                      !isStepRunning && "hover:bg-slate-50",
                      isOpen ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0"
                    )}
                    style={{
                      transitionDelay: isOpen ? `${index * 60}ms` : "0ms",
                    }}
                    title={detail}
                    aria-expanded={isExpanded}
                    onClick={(event) => {
                      event.stopPropagation()
                      setExpandedIndex((current) => (current === index ? null : index))
                    }}
                  >
                    <div
                      className={cn(
                        "flex h-9 w-9 shrink-0 items-center justify-center rounded-xl transition-colors duration-300",
                        isStepRunning && "bg-blue-100",
                        isCompleted && "bg-emerald-50",
                        isSkipped && "bg-slate-100"
                      )}
                    >
                      {isStepRunning ? (
                        <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
                      ) : isCompleted ? (
                        <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                      ) : (
                        <SkipForward className="h-4 w-4 text-slate-400" />
                      )}
                    </div>

                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5">
                        <NodeIcon
                          className={cn(
                            "h-3.5 w-3.5 shrink-0",
                            isStepRunning && "text-blue-600",
                            isCompleted && "text-slate-500",
                            isSkipped && "text-slate-400"
                          )}
                        />
                        <h4
                          className={cn(
                            "text-sm font-medium",
                            isStepRunning && "text-blue-900",
                            isCompleted && "text-slate-800",
                            isSkipped && "text-slate-500"
                          )}
                        >
                          {config.label}
                        </h4>
                      </div>
                      {step.detail && (
                        <p
                          className={cn(
                            "mt-0.5 text-xs leading-5 text-slate-500",
                            isExpanded ? "whitespace-normal break-words" : "truncate"
                          )}
                        >
                          {detail}
                        </p>
                      )}
                    </div>

                    <span className="shrink-0 pt-1 font-mono text-[10px] text-slate-400">
                      {formatDuration(step.duration_ms)}
                    </span>
                  </button>
                )
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
