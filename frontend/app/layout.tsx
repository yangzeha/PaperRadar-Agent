import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
})

export const metadata: Metadata = {
  title: "PaperRadar-Agent - 中文论文雷达",
  description:
    "基于 LangGraph 的中文论文雷达与选题追踪 Agent，支持论文检索、RAG、长期记忆和国内模型切换。",
  keywords: ["PaperRadar", "Agent", "LangGraph", "RAG", "论文雷达", "选题追踪"],
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="zh-CN">
      <body className={`${inter.variable} bg-white text-slate-950 font-sans antialiased`}>
        {children}
      </body>
    </html>
  )
}
