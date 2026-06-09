#!/usr/bin/env node

const assertContains = (html, text, failures) => {
  if (!html.includes(text)) {
    failures.push(`页面缺少：${text}`)
  }
}

const assertNotContains = (html, text, failures) => {
  if (html.includes(text)) {
    failures.push(`页面不应出现：${text}`)
  }
}

async function main() {
  const baseUrl = process.env.FRONTEND_BASE_URL || "http://localhost:3000"
  const response = await fetch(baseUrl)
  if (!response.ok) {
    throw new Error(`Homepage request failed: HTTP ${response.status}`)
  }

  const html = await response.text()
  const failures = []

  const required = [
    "功能介绍",
    "PaperRadar-Agent 功能介绍",
    "研究方向雷达",
    "代表论文推荐",
  ]

  const forbidden = [
    "Agentic RAG 方向论文雷达",
    "LLM Agent 长期记忆",
    "Multi-Agent Collaboration",
    "RAG Hallucination Evaluation",
    "mock/fallback 模式",
    "论文数量一致性检查器",
    "年份过滤核查器",
    "摘要证据表",
  ]

  for (const text of required) {
    assertContains(html, text, failures)
  }
  for (const text of forbidden) {
    assertNotContains(html, text, failures)
  }

  if (failures.length > 0) {
    console.error("Homepage DOM check failed:")
    for (const failure of failures) {
      console.error(`- ${failure}`)
    }
    console.error("\nHomepage preview:")
    console.error(html.slice(0, 2000))
    process.exit(1)
  }

  console.log("Homepage DOM check passed.")
  console.log(`url=${baseUrl}`)
  console.log("required=功能介绍, PaperRadar-Agent 功能介绍, 研究方向雷达, 代表论文推荐")
  console.log("forbidden_absent=old recommendation buttons, mock/fallback, old report sections")
}

main().catch((error) => {
  console.error(`Homepage DOM check failed: ${error.message}`)
  process.exit(1)
})
