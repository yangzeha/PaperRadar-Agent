"use client"

import React, { useState, useRef, useEffect, useCallback, forwardRef } from "react"
import * as TooltipPrimitive from "@radix-ui/react-tooltip"
import {
  ArrowUp,
  Search,
  BookOpen,
  FlaskConical,
  Settings2,
  Globe2,
  RadioTower,
} from "lucide-react"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"
import type { SearchRequest } from "@/lib/types"

// --- Textarea Component ---
interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  className?: string
}
const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => (
    <textarea
      className={cn(
        "flex w-full rounded-md border-none bg-transparent px-3 py-2.5 text-base text-slate-900 placeholder:text-slate-400 focus-visible:outline-none focus-visible:ring-0 disabled:cursor-not-allowed disabled:opacity-50 min-h-[44px] resize-none",
        className
      )}
      ref={ref}
      rows={1}
      {...props}
    />
  )
)
Textarea.displayName = "Textarea"

// --- Tooltip Components ---
const TooltipProvider = TooltipPrimitive.Provider
const Tooltip = TooltipPrimitive.Root
const TooltipTrigger = TooltipPrimitive.Trigger
const TooltipContent = forwardRef<
  React.ElementRef<typeof TooltipPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Content>
>(({ className, sideOffset = 4, ...props }, ref) => (
  <TooltipPrimitive.Content
    ref={ref}
    sideOffset={sideOffset}
    className={cn(
      "z-50 overflow-hidden rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-700 shadow-md animate-in fade-in-0 zoom-in-95",
      className
    )}
    {...props}
  />
))
TooltipContent.displayName = TooltipPrimitive.Content.displayName

// --- Custom Divider ---
const CustomDivider: React.FC = () => (
  <div className="relative h-6 w-[1.5px] mx-1">
    <div
      className="absolute inset-0 bg-gradient-to-t from-transparent via-[#9b87f5]/70 to-transparent rounded-full"
      style={{
        clipPath:
          "polygon(0% 0%, 100% 0%, 100% 40%, 140% 50%, 100% 60%, 100% 100%, 0% 100%, 0% 60%, -40% 50%, 0% 40%)",
      }}
    />
  </div>
)

// --- PromptInput Context ---
interface PromptInputContextType {
  isLoading: boolean
  value: string
  setValue: (value: string) => void
  maxHeight: number | string
  onSubmit?: () => void
  disabled?: boolean
}
const PromptInputContext = React.createContext<PromptInputContextType>({
  isLoading: false,
  value: "",
  setValue: () => {},
  maxHeight: 240,
  onSubmit: undefined,
  disabled: false,
})
function usePromptInput() {
  return React.useContext(PromptInputContext)
}

// --- PromptInput ---
interface PromptInputProps {
  isLoading?: boolean
  value?: string
  onValueChange?: (value: string) => void
  maxHeight?: number | string
  onSubmit?: () => void
  children: React.ReactNode
  className?: string
  disabled?: boolean
}
const PromptInput = forwardRef<HTMLDivElement, PromptInputProps>(
  (
    {
      className,
      isLoading = false,
      maxHeight = 240,
      value,
      onValueChange,
      onSubmit,
      children,
      disabled = false,
    },
    ref
  ) => {
    const [internalValue, setInternalValue] = useState(value || "")
    const handleChange = (newValue: string) => {
      setInternalValue(newValue)
      onValueChange?.(newValue)
    }
    return (
      <TooltipProvider>
        <PromptInputContext.Provider
          value={{
            isLoading,
            value: value ?? internalValue,
            setValue: onValueChange ?? handleChange,
            maxHeight,
            onSubmit,
            disabled,
          }}
        >
          <div
            ref={ref}
            className={cn(
              "rounded-3xl border border-slate-200 bg-white p-2 shadow-[0_18px_45px_rgba(15,23,42,0.10)] transition-all duration-300",
              className
            )}
          >
            {children}
          </div>
        </PromptInputContext.Provider>
      </TooltipProvider>
    )
  }
)
PromptInput.displayName = "PromptInput"

// --- PromptInputTextarea ---
const PromptInputTextarea: React.FC<
  { disableAutosize?: boolean; placeholder?: string } & React.ComponentProps<typeof Textarea>
> = ({ className, onKeyDown, disableAutosize = false, placeholder, ...props }) => {
  const { value, setValue, maxHeight, onSubmit, disabled } = usePromptInput()
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (disableAutosize || !textareaRef.current) return
    textareaRef.current.style.height = "auto"
    textareaRef.current.style.height =
      typeof maxHeight === "number"
        ? `${Math.min(textareaRef.current.scrollHeight, maxHeight)}px`
        : `min(${textareaRef.current.scrollHeight}px, ${maxHeight})`
  }, [value, maxHeight, disableAutosize])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      onSubmit?.()
    }
    onKeyDown?.(e)
  }

  return (
    <Textarea
      ref={textareaRef}
      value={value}
      onChange={(e) => setValue(e.target.value)}
      onKeyDown={handleKeyDown}
      className={cn("text-base", className)}
      disabled={disabled}
      placeholder={placeholder}
      {...props}
    />
  )
}

// --- Source Toggle Button ---
interface SourceToggleProps {
  active: boolean
  onClick: () => void
  icon: React.ReactNode
  label: string
  activeColor: string
  activeBg: string
}
const SourceToggle: React.FC<SourceToggleProps> = ({
  active,
  onClick,
  icon,
  label,
  activeColor,
  activeBg,
}) => (
  <button
    type="button"
    onClick={onClick}
    className={cn(
      "rounded-full transition-all flex items-center gap-1 px-2 py-1 border h-8",
      active
        ? `${activeBg} border-current ${activeColor}`
        : "bg-transparent border-transparent text-slate-500 hover:text-slate-800"
    )}
  >
    <div className="w-5 h-5 flex items-center justify-center flex-shrink-0">
      <motion.div
        animate={{
          rotate: active ? 360 : 0,
          scale: active ? 1.1 : 1,
        }}
        whileHover={{
          rotate: active ? 360 : 15,
          scale: 1.1,
          transition: { type: "spring", stiffness: 300, damping: 10 },
        }}
        transition={{ type: "spring", stiffness: 260, damping: 25 }}
      >
        {icon}
      </motion.div>
    </div>
    <span className="text-xs whitespace-nowrap flex-shrink-0">
      {label}
    </span>
  </button>
)

// --- Main SearchBar Component ---
interface SearchBarProps {
  onSearch: (request: SearchRequest) => void
  isLoading: boolean
}

export function SearchBar({ onSearch, isLoading }: SearchBarProps) {
  const [query, setQuery] = useState("")
  const [arxivActive, setArxivActive] = useState(true)
  const [pubmedActive, setPubmedActive] = useState(true)
  const [openAlexActive, setOpenAlexActive] = useState(true)
  const [ieeeActive, setIeeeActive] = useState(false)
  const [maxResults, setMaxResults] = useState("10")

  const activeSourceCount =
    Number(arxivActive) +
    Number(pubmedActive) +
    Number(openAlexActive) +
    Number(ieeeActive)

  const toggleArxiv = useCallback(() => {
    setArxivActive((prev) => {
      if (!prev) return true
      if (activeSourceCount <= 1) return true
      return false
    })
  }, [activeSourceCount])

  const togglePubmed = useCallback(() => {
    setPubmedActive((prev) => {
      if (!prev) return true
      if (activeSourceCount <= 1) return true
      return false
    })
  }, [activeSourceCount])

  const toggleOpenAlex = useCallback(() => {
    setOpenAlexActive((prev) => {
      if (!prev) return true
      if (activeSourceCount <= 1) return true
      return false
    })
  }, [activeSourceCount])

  const toggleIeee = useCallback(() => {
    setIeeeActive((prev) => {
      if (!prev) return true
      if (activeSourceCount <= 1) return true
      return false
    })
  }, [activeSourceCount])

  const handleSubmit = useCallback(() => {
    const trimmedQuery = query.trim()
    if (!trimmedQuery || isLoading) return
    const sources: string[] = []
    if (arxivActive) sources.push("arxiv")
    if (pubmedActive) sources.push("pubmed")
    if (openAlexActive) sources.push("openalex")
    if (ieeeActive) sources.push("ieee")
    const requestedResults = Math.max(
      1,
      Math.min(Number.parseInt(maxResults, 10) || 10, 99)
    )
    setQuery("")
    onSearch({
      query: trimmedQuery,
      sources,
      max_results: requestedResults,
    })
  }, [
    query,
    isLoading,
    arxivActive,
    pubmedActive,
    openAlexActive,
    ieeeActive,
    maxResults,
    onSearch,
  ])

  const hasContent = query.trim() !== ""

  return (
    <div className="w-full max-w-3xl mx-auto">
      <PromptInput
        value={query}
        onValueChange={setQuery}
        isLoading={isLoading}
        onSubmit={handleSubmit}
        className="w-full bg-white border-slate-200 shadow-[0_18px_45px_rgba(15,23,42,0.10)] transition-all duration-300 ease-in-out"
        disabled={isLoading}
      >
        <PromptInputTextarea
          placeholder="输入研究方向，例如：大模型智能体记忆机制"
          className="text-base"
        />

        <div className="flex flex-wrap items-center justify-between gap-2 p-0 pt-2">
          {/* Left: Source toggles */}
          <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1">
            <SourceToggle
              active={arxivActive}
              onClick={toggleArxiv}
              icon={
                <BookOpen
                  className={cn(
                    "w-4 h-4",
                    arxivActive ? "text-[#3B82F6]" : "text-inherit"
                  )}
                />
              }
              label="arXiv"
              activeColor="text-[#3B82F6]"
              activeBg="bg-[#3B82F6]/15"
            />

            <CustomDivider />

            <SourceToggle
              active={pubmedActive}
              onClick={togglePubmed}
              icon={
                <FlaskConical
                  className={cn(
                    "w-4 h-4",
                    pubmedActive ? "text-[#10B981]" : "text-inherit"
                  )}
                />
              }
              label="PubMed"
              activeColor="text-[#10B981]"
              activeBg="bg-[#10B981]/15"
            />

            <CustomDivider />

            <SourceToggle
              active={openAlexActive}
              onClick={toggleOpenAlex}
              icon={
                <Globe2
                  className={cn(
                    "w-4 h-4",
                    openAlexActive ? "text-[#8B5CF6]" : "text-inherit"
                  )}
                />
              }
              label="OpenAlex"
              activeColor="text-[#8B5CF6]"
              activeBg="bg-[#8B5CF6]/15"
            />

            <CustomDivider />

            <SourceToggle
              active={ieeeActive}
              onClick={toggleIeee}
              icon={
                <RadioTower
                  className={cn(
                    "w-4 h-4",
                    ieeeActive ? "text-[#EF4444]" : "text-inherit"
                  )}
                />
              }
              label="IEEE"
              activeColor="text-[#EF4444]"
              activeBg="bg-[#EF4444]/15"
            />

            <CustomDivider />

            {/* Max results input */}
            <div className="flex h-8 items-center gap-1.5 rounded-full border border-[#F59E0B]/40 bg-[#F59E0B]/10 px-2 text-[#B45309]">
              <div className="w-5 h-5 flex items-center justify-center">
                <Settings2 className="w-4 h-4" />
              </div>
              <span className="text-xs whitespace-nowrap">数量:</span>
              <input
                type="number"
                min={1}
                max={99}
                value={maxResults}
                onChange={(e) => {
                  const value = e.target.value
                  if (value === "") {
                    setMaxResults("")
                    return
                  }
                  const next = Math.max(
                    1,
                    Math.min(Number.parseInt(value, 10) || 1, 99)
                  )
                  setMaxResults(String(next))
                }}
                onBlur={() => {
                  if (!maxResults) setMaxResults("10")
                }}
                disabled={isLoading}
                className="h-6 w-14 rounded-md border border-slate-200 bg-white px-2 text-sm text-slate-800 outline-none focus:border-[#F59E0B] disabled:cursor-not-allowed disabled:opacity-60"
              />
            </div>
          </div>

          {/* Right: Submit button */}
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={handleSubmit}
                disabled={!hasContent || isLoading}
                className={cn(
                  "h-8 w-8 rounded-full inline-flex items-center justify-center transition-all duration-200",
                  hasContent && !isLoading
                    ? "bg-slate-950 hover:bg-slate-800 text-white cursor-pointer"
                    : "bg-transparent text-slate-400 cursor-not-allowed opacity-50"
                )}
              >
                {isLoading ? (
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{
                      duration: 1,
                      repeat: Infinity,
                      ease: "linear",
                    }}
                  >
                    <Search className="h-4 w-4" />
                  </motion.div>
                ) : (
                  <ArrowUp className="h-4 w-4" />
                )}
              </button>
            </TooltipTrigger>
            <TooltipContent side="top">
              {isLoading ? "正在检索..." : "生成论文雷达"}
            </TooltipContent>
          </Tooltip>
        </div>
      </PromptInput>
    </div>
  )
}
