"use client"

import { motion } from "framer-motion"
import { ExternalLink, Workflow } from "lucide-react"
import { AccountSettingsButton } from "@/components/account-settings"

const LANGGRAPH_STUDIO_URL =
  "https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2025"

export function Header() {
  return (
    <header className="w-full z-50 relative">
      <div className="container flex h-14 items-center justify-between px-4 md:px-8">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="flex items-center gap-2.5"
        >
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-950">
            <span className="text-base font-black text-white tracking-tighter leading-none">P</span>
          </div>
          <span className="text-base font-bold tracking-tight text-slate-950">
            PaperRadar-Agent
          </span>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="flex items-center gap-3"
        >
          <div className="hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/80 border border-slate-200 shadow-sm">
            <div className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs text-slate-600">LangGraph + 国内模型可切换</span>
          </div>
          <a
            href={LANGGRAPH_STUDIO_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex h-9 items-center gap-1.5 rounded-full border border-blue-200 bg-blue-50 px-3 text-sm font-medium text-blue-700 shadow-sm transition-colors hover:border-blue-300 hover:bg-blue-100"
            title="打开 LangGraph Studio 查看节点和状态"
          >
            <Workflow className="h-4 w-4" />
            <span className="hidden sm:inline">Studio</span>
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
          <AccountSettingsButton />
        </motion.div>
      </div>
    </header>
  )
}
