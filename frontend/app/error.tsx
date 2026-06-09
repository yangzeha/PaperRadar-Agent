"use client"

import { useEffect } from "react"
import { AlertTriangle, RotateCcw } from "lucide-react"

interface ErrorPageProps {
  error: Error & { digest?: string }
  reset: () => void
}

export default function ErrorPage({ error, reset }: ErrorPageProps) {
  useEffect(() => {
    console.error("PaperRadar page error:", error)
  }, [error])

  return (
    <div className="flex min-h-screen items-center justify-center bg-white px-4 text-slate-900">
      <div className="w-full max-w-md rounded-2xl border border-red-100 bg-white p-6 text-center shadow-sm">
        <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-red-50">
          <AlertTriangle className="h-5 w-5 text-red-600" />
        </div>
        <h1 className="text-lg font-semibold">页面刚才出现了一个渲染错误</h1>
        <p className="mt-2 text-sm leading-6 text-slate-500">
          我已经为历史会话和论文数据增加了兼容处理。你可以重新加载当前页面继续使用。
        </p>
        <button
          type="button"
          onClick={reset}
          className="mt-5 inline-flex h-10 items-center gap-2 rounded-lg bg-slate-950 px-4 text-sm font-medium text-white transition-colors hover:bg-slate-800"
        >
          <RotateCcw className="h-4 w-4" />
          重新加载
        </button>
      </div>
    </div>
  )
}
