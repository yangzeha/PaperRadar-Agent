"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { BookOpen, ChevronDown, ChevronUp } from "lucide-react"
import { PaperCard } from "@/components/paper-card"
import type { PaperResult } from "@/lib/types"

interface SourceListProps {
  papers: PaperResult[]
}

export function SourceList({ papers = [] }: SourceListProps) {
  const [expanded, setExpanded] = useState(false)
  const safePapers = Array.isArray(papers) ? papers : []

  if (safePapers.length === 0) return null

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4, delay: 0.2 }}
      className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm"
    >
      <button
        type="button"
        onClick={() => setExpanded((value) => !value)}
        className="flex w-full items-center gap-2 text-left"
      >
        <BookOpen className="h-4 w-4 text-blue-600" />
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold text-slate-900">
            检索到的论文库（{safePapers.length} 篇）
          </h3>
          <p className="mt-0.5 text-xs text-slate-500">
            默认收起，展开查看摘要、来源和论文角色
          </p>
        </div>
        {expanded ? (
          <ChevronUp className="h-4 w-4 text-slate-500" />
        ) : (
          <ChevronDown className="h-4 w-4 text-slate-500" />
        )}
      </button>

      {expanded && (
        <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-2">
          {safePapers.map((paper, index) => (
            <PaperCard key={`${paper.url}-${index}`} paper={paper} index={index} />
          ))}
        </div>
      )}
    </motion.div>
  )
}
