# PaperRadar 模式说明

## 项目定位

PaperRadar-Agent 是一个面向 Agent 实习岗位的论文雷达项目。它不是只做“搜索后总结”，也不是把论文标题排成列表，而是把论文检索、向量召回、相关性评分、路线分类、结构化生成、幻觉检查、引用整理和长期记忆串成一个 LangGraph Agentic RAG 工作流。

## 新增能力

1. 中文 PaperRadar 报告
   - 固定输出 8 个章节：方向概览、方法路线分类、代表论文推荐、近年趋势、研究空白、两周阅读路线、可做小项目建议、参考来源。
   - 方法路线分类必须输出表格，路线包括 Survey / Taxonomy、Planning / Reasoning、Multi-Agent / Hierarchical、Multimodal RAG、Evaluation / Benchmark、Domain-specific Agentic RAG 等实际命中的类别。
   - 研究空白必须给出具体 gap、可验证方式和两周内可做 demo。
   - 报告用 `[1]`、`[2]` 引用论文来源。

2. 任务路由
   - `paper_search`：普通论文搜索。
   - `paper_qa`：基于论文回答问题。
   - `paper_radar`：生成中文方向雷达报告。
   - `reading_plan`：偏两周阅读路线。
   - `project_idea`：偏可落地小项目建议。
   - `general`：普通对话，不进入检索链路。

3. 长期记忆
   - `backend/data/memory/user_topics.json`：用户长期关注方向。
   - `backend/data/memory/saved_papers.json`：待读论文。
   - `backend/data/memory/reading_history.json`：检索和报告历史。
   - 缺文件或 JSON 损坏时会自动降级为空数据。

4. 国内模型适配
   - `LLM_PROVIDER=deepseek`：调用 DeepSeek OpenAI-compatible API。
   - `LLM_PROVIDER=qwen`：调用 Qwen/DashScope OpenAI-compatible API。
   - `LLM_PROVIDER=mock`：不出网，稳定输出，用于测试和演示。

## 报告质量验收

mock 模式和 real 模式都必须遵守同一报告质量标准。报告至少要满足：

- 包含 8 个固定章节。
- 至少 2 个 Markdown 表格：方法路线分类表、代表论文推荐表或两周阅读路线表。
- 至少 5 个具体研究空白，使用 `Gap 1` / `Gap 2` 结构。
- 至少 3 个 Agentic RAG 强相关小项目。
- 至少 5 个引用标记，例如 `[1]`。
- 不出现 `mock`、`fallback` 等内部实现词。
- 小项目不能只写“论文数量一致性检查器”“年份过滤核查器”这类内部校验工具。

运行质量 smoke：

```powershell
cd backend
python scripts/smoke_report_quality.py
```

## 三个演示问题

### 示例 1：Agentic RAG

```text
Agentic RAG 方向论文雷达：趋势、代表论文、研究空白和两周阅读路线
```

适合展示 Router 将任务分到 `paper_radar`，Retriever 检索论文，Grader 过滤相关论文，Generator 输出 8 节中文报告。

### 示例 2：长期记忆

```text
LLM Agent 长期记忆机制论文雷达
```

适合展示长期记忆如何记录用户 topic，并在下一次报告中作为 memory context 放入 prompt。

### 示例 3：幻觉评估

```text
RAG Hallucination Evaluation 的研究趋势、代表论文和小项目建议
```

适合展示 Hallucination Checker 如何对生成答案做 groundedness 分数检查。

## 两周学习路线

- 第 1-2 天：读 `backend/app/agents/graph.py`，理解 LangGraph 节点和条件边。
- 第 3-4 天：读 `router.py`、`retriever.py`、`grader.py`，理解任务分类和 RAG 召回。
- 第 5-6 天：读 `generator.py`、`hallucination_checker.py`、`synthesizer.py`，理解生成、核查、引用清理。
- 第 7-8 天：读 `memory_store.py` 和 FastAPI 记忆接口。
- 第 9-10 天：切换 `LLM_PROVIDER=mock/deepseek/qwen`，跑通不同 Provider。
- 第 11-12 天：读前端 `search-bar.tsx`、`answer-panel.tsx`、`thinking-steps.tsx`。
- 第 13-14 天：整理 README、截图、测试结果和简历 STAR 描述。

## 简历表述参考

```text
PaperRadar-Agent：基于 LangGraph 的中文论文雷达与选题追踪 Agent。
二次开发 ScholarAgent，保留 arXiv/PubMed 检索、ChromaDB 向量检索和 Agentic RAG 图流程；
新增 paper_radar/reading_plan/project_idea 任务路由、中文 8 节结构化报告、JSON 长期记忆、DeepSeek/Qwen OpenAI-compatible Provider 和 mock 测试模式。
通过 Router/Retriever/Grader/Rewriter/Generator/HallucinationChecker/Synthesizer 完成可追踪论文检索、相关性过滤、引用生成和 groundedness 检查。
```
