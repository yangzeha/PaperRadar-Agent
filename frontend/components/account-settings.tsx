"use client"

import { useEffect, useMemo, useState } from "react"
import { Check, Settings, UserCircle, X } from "lucide-react"
import { cn } from "@/lib/utils"

type AccountSettings = {
  name: string
  email: string
  role: string
  organization: string
  researchFocus: string
}

const STORAGE_KEY = "paper-radar-account-settings"

const DEFAULT_SETTINGS: AccountSettings = {
  name: "研究者",
  email: "",
  role: "学生 / 求职者",
  organization: "",
  researchFocus: "LLM Agent, RAG, 长期记忆",
}

export function AccountSettingsButton() {
  const [isOpen, setIsOpen] = useState(false)
  const [settings, setSettings] = useState<AccountSettings>(DEFAULT_SETTINGS)
  const [draft, setDraft] = useState<AccountSettings>(DEFAULT_SETTINGS)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY)
      if (!raw) return
      const parsed = JSON.parse(raw) as Partial<AccountSettings>
      const next = { ...DEFAULT_SETTINGS, ...parsed }
      setSettings(next)
      setDraft(next)
    } catch {
      setSettings(DEFAULT_SETTINGS)
      setDraft(DEFAULT_SETTINGS)
    }
  }, [])

  const initials = useMemo(() => {
    const name = settings.name.trim() || DEFAULT_SETTINGS.name
    return name.slice(0, 1).toUpperCase()
  }, [settings.name])

  function updateDraft(key: keyof AccountSettings, value: string) {
    setSaved(false)
    setDraft((prev) => ({ ...prev, [key]: value }))
  }

  function saveSettings() {
    const next = {
      ...draft,
      name: draft.name.trim() || DEFAULT_SETTINGS.name,
    }
    setSettings(next)
    setDraft(next)
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
    setSaved(true)
  }

  function closePanel() {
    setDraft(settings)
    setSaved(false)
    setIsOpen(false)
  }

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setIsOpen((open) => !open)}
        className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-2.5 py-1.5 text-sm font-medium text-slate-700 shadow-sm transition-colors hover:border-blue-300 hover:text-blue-700"
        aria-label="打开账号设置"
      >
        <span className="flex h-7 w-7 items-center justify-center rounded-full bg-blue-600 text-xs font-bold text-white">
          {initials}
        </span>
        <span className="hidden max-w-28 truncate sm:inline">{settings.name}</span>
        <Settings className="h-4 w-4 text-slate-400" />
      </button>

      {isOpen && (
        <div className="absolute right-0 top-12 z-50 w-[min(92vw,360px)] overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl shadow-slate-900/15">
          <div className="flex items-start justify-between border-b border-slate-100 px-4 py-3">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-50 text-blue-600">
                <UserCircle className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-sm font-semibold text-slate-950">账号设置</h2>
                <p className="text-xs text-slate-500">保存你的研究身份和偏好</p>
              </div>
            </div>
            <button
              type="button"
              onClick={closePanel}
              className="rounded-full p-1.5 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-700"
              aria-label="关闭账号设置"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <div className="space-y-3 px-4 py-4">
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-slate-600">昵称</span>
              <input
                value={draft.name}
                onChange={(event) => updateDraft("name", event.target.value)}
                className="h-10 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-900 outline-none transition-colors placeholder:text-slate-400 focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
                placeholder="例如：一名 Agent 方向研究者"
              />
            </label>

            <label className="block">
              <span className="mb-1 block text-xs font-medium text-slate-600">邮箱</span>
              <input
                value={draft.email}
                onChange={(event) => updateDraft("email", event.target.value)}
                type="email"
                className="h-10 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-900 outline-none transition-colors placeholder:text-slate-400 focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
                placeholder="name@example.com"
              />
            </label>

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <label className="block">
                <span className="mb-1 block text-xs font-medium text-slate-600">身份</span>
                <input
                  value={draft.role}
                  onChange={(event) => updateDraft("role", event.target.value)}
                  className="h-10 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-900 outline-none transition-colors placeholder:text-slate-400 focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
                  placeholder="学生 / 工程师"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-xs font-medium text-slate-600">学校/机构</span>
                <input
                  value={draft.organization}
                  onChange={(event) => updateDraft("organization", event.target.value)}
                  className="h-10 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-900 outline-none transition-colors placeholder:text-slate-400 focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
                  placeholder="可选"
                />
              </label>
            </div>

            <label className="block">
              <span className="mb-1 block text-xs font-medium text-slate-600">研究方向</span>
              <textarea
                value={draft.researchFocus}
                onChange={(event) => updateDraft("researchFocus", event.target.value)}
                className="min-h-20 w-full resize-none rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition-colors placeholder:text-slate-400 focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
                placeholder="例如：Agentic RAG, LLM Agent 记忆, 幻觉评估"
              />
            </label>
          </div>

          <div className="flex items-center justify-between border-t border-slate-100 bg-slate-50 px-4 py-3">
            <span
              className={cn(
                "inline-flex items-center gap-1 text-xs transition-opacity",
                saved ? "text-emerald-600 opacity-100" : "text-slate-400 opacity-0"
              )}
            >
              <Check className="h-3.5 w-3.5" />
              已保存
            </span>
            <button
              type="button"
              onClick={saveSettings}
              className="inline-flex h-9 items-center justify-center rounded-lg bg-blue-600 px-3 text-sm font-medium text-white shadow-sm transition-colors hover:bg-blue-700"
            >
              保存设置
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
