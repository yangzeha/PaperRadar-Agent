"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { ExternalLink, ChevronDown, ChevronUp, Calendar } from "lucide-react"
import { cn } from "@/lib/utils"
import type { PaperResult } from "@/lib/types"

interface PaperCardProps {
  paper: PaperResult
  index: number
}

export function PaperCard({ paper, index }: PaperCardProps) {
  const [expanded, setExpanded] = useState(false)

  const MAX_AUTHORS = 3
  const authors = Array.isArray(paper.authors) ? paper.authors : []
  const abstract = typeof paper.abstract === "string" ? paper.abstract : ""
  const displayAuthors = authors.slice(0, MAX_AUTHORS)
  const remainingAuthors = authors.length - MAX_AUTHORS

  const ABSTRACT_LENGTH = 200
  const isLongAbstract = abstract.length > ABSTRACT_LENGTH
  const displayAbstract = expanded
    ? abstract
    : abstract.slice(0, ABSTRACT_LENGTH) +
      (isLongAbstract ? "..." : "")

  const formattedDate = paper.published
    ? new Date(paper.published).toLocaleDateString("zh-CN", {
        year: "numeric",
        month: "short",
        day: "numeric",
      })
    : null

  const sourceLabel =
    paper.source === "arxiv"
      ? "arXiv"
      : paper.source === "pubmed"
        ? "PubMed"
        : paper.source === "ieee"
          ? "IEEE"
          : "OpenAlex"

  const roleLabel =
    paper.role_label ||
    (paper.role === "core_gcl_method"
      ? "核心方法"
      : paper.role === "overview_survey"
        ? "综述"
        : paper.role === "background_related"
          ? "相关但非核心"
          : paper.is_background
            ? "背景"
            : paper.is_core
              ? "核心论文"
              : null)

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.08 }}
      className="group rounded-xl border border-slate-200 bg-white shadow-sm transition-all duration-300 hover:border-blue-200 hover:shadow-md"
    >
      <div className="p-4">
        {/* Header */}
        <div className="flex items-start justify-between gap-2 mb-2">
          <a
            href={paper.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm font-semibold text-slate-900 hover:text-blue-700 transition-colors flex items-start gap-1.5 flex-1"
          >
            <span className="flex-1">{paper.title}</span>
            <ExternalLink className="h-3.5 w-3.5 mt-0.5 shrink-0 opacity-0 group-hover:opacity-70 transition-opacity" />
          </a>
          <div className="flex shrink-0 flex-col items-end gap-1">
            <span
              className={cn(
                "text-[10px] font-medium px-2 py-0.5 rounded-full border",
                paper.source === "arxiv"
                  ? "bg-blue-50 text-blue-700 border-blue-200"
                  : paper.source === "pubmed"
                    ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                    : paper.source === "ieee"
                      ? "bg-red-50 text-red-700 border-red-200"
                      : "bg-violet-50 text-violet-700 border-violet-200"
              )}
            >
              {sourceLabel}
            </span>
            {roleLabel && (
              <span className="rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[10px] font-medium text-amber-700">
                {roleLabel}
              </span>
            )}
          </div>
        </div>

        {/* Authors */}
        <p className="text-xs text-slate-500 mb-2">
          {displayAuthors.join(", ")}
          {remainingAuthors > 0 && (
            <span className="text-slate-400">
              {" "}
              +{remainingAuthors} 位作者
            </span>
          )}
        </p>

        {/* Abstract */}
        <p className="text-xs text-slate-600 leading-relaxed mb-2">
          {displayAbstract}
        </p>

        {isLongAbstract && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="inline-flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700 transition-colors mb-2"
          >
            {expanded ? (
              <>
                收起 <ChevronUp className="h-3 w-3" />
              </>
            ) : (
              <>
                展开摘要 <ChevronDown className="h-3 w-3" />
              </>
            )}
          </button>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between pt-2 border-t border-slate-100">
          {formattedDate && (
            <div className="flex items-center gap-1 text-[10px] text-slate-500">
              <Calendar className="h-3 w-3" />
              {formattedDate}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  )
}
