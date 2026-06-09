# PaperRadar-Agent

> 基于 LangGraph 的中文论文雷达与选题追踪 Agent：输入一个研究方向，系统会自动检索论文、筛选相关文献、生成结构化中文报告，并给出引用、研究空白、两周阅读路线和可落地小项目建议。

PaperRadar-Agent 不是一个简单的“论文搜索 + 总结”页面，而是一个可观测的 Agentic RAG 项目。它把论文调研拆成多个可追踪节点：任务路由、论文检索、相关性评分、查询改写、报告生成、幻觉检查和引用整理。这个设计既适合真实使用，也适合作为简历项目向面试官讲清楚 Agent/RAG 工程能力。

## 项目预览

### 前端界面

![PaperRadar 前端界面](docs/screenshot.png)

### Agent 工作流

![LangGraph Agent 工作流](exported_image.png)

## 项目能做什么

用户输入一个方向，例如：

```text
Agentic RAG 方向论文雷达：趋势、代表论文、研究空白和两周阅读路线
```

系统会输出一份中文 PaperRadar 报告，通常包括：

- 方向概览：解释这个研究方向是什么、为什么重要、核心问题在哪里。
- 方法路线分类：把检索到的论文按 Survey、Planning、Multi-Agent、Evaluation、Domain-specific 等路线组织。
- 代表论文推荐：按必读、重点、背景分层说明哪些论文值得先看。
- 近年趋势：按年份或阶段总结技术演进。
- 研究空白：给出可验证的 gap，而不是泛泛而谈。
- 两周阅读路线：把论文阅读拆成阶段目标和产出。
- 小项目建议：给出能写进简历的 MVP 方案。
- 参考来源：用 `[1]`、`[2]` 等引用对齐论文来源。

## 核心功能

- Agentic RAG 工作流：用 LangGraph 把复杂论文调研拆成多个节点，每一步都有状态记录。
- 多源论文检索：支持 arXiv、PubMed、OpenAlex，配置 IEEE API key 后可扩展到 IEEE Xplore。
- 查询路由：区分普通对话、论文搜索、论文雷达、阅读计划、小项目建议等任务类型。
- 相关性评分：Retriever 先找候选论文，Grader 再筛出真正相关的文献。
- 查询改写：如果检索结果不理想，Rewriter 会改写 query 后重新检索。
- 中文结构化生成：Generator 按固定报告结构输出适合学习和面试展示的中文调研报告。
- 幻觉检测：Hallucination Checker 会给出 `0.0 - 1.0` 的幻觉分数，分数越高代表越不可信。
- 引用清理：Synthesizer 会清理无效引用，保证正文中的 `[1]` 能对应到真实来源。
- 长期记忆：保存用户关注主题、待读论文、历史检索和聊天会话摘要。
- 前端可视化：展示聊天历史、执行步骤、论文卡片、引用来源和生成报告。

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 后端 API | FastAPI、Pydantic、Uvicorn |
| Agent 编排 | LangGraph、LangChain Core |
| LLM Provider | mock、DeepSeek、Qwen/DashScope、Gemini |
| 论文检索 | arXiv、PubMed E-utilities、OpenAlex、IEEE Xplore |
| 向量检索 | ChromaDB、sentence-transformers |
| 前端 | Next.js、React、Tailwind CSS、Framer Motion、lucide-react |
| 存储 | 本地 JSON 长期记忆、Chroma 持久化向量库 |
| 测试 | pytest、respx、FastAPI TestClient |
| 部署 | Docker、Docker Compose |

## 系统架构

项目分为四层：

1. 前端层：Next.js 页面负责输入问题、展示报告、论文卡片、引用来源和执行步骤。
2. API 层：FastAPI 提供 `/api/search`、`/api/translate`、`/api/chat/sessions`、`/api/memory/*` 等接口。
3. Agent 层：LangGraph 负责组织 Router、Retriever、Grader、Rewriter、Generator、Hallucination Checker、Synthesizer。
4. 服务层：封装 LLM Provider、论文检索、向量库、长期记忆、论文分类和角色分配。

核心链路如下：

```text
前端输入
  -> FastAPI /api/search
  -> LangGraph 初始化 AgentState
  -> Router 判断任务类型
  -> Retriever 检索论文
  -> Grader 过滤相关论文
  -> Rewriter 在结果不好时改写查询
  -> Generator 生成中文报告
  -> Hallucination Checker 检查答案是否 grounded
  -> Synthesizer 整理引用和最终输出
  -> 前端展示报告、论文和执行步骤
```

## LangGraph 节点说明

| 节点 | 作用 |
| --- | --- |
| Router | 判断用户请求属于普通对话、论文搜索、论文雷达、阅读计划还是项目建议。 |
| Retriever | 根据 query 从 arXiv、PubMed、OpenAlex 等来源检索论文。 |
| Grader | 判断检索到的论文是否真正和用户问题相关。 |
| Rewriter | 当相关文档不足时，改写 query 并重新进入检索流程。 |
| Generator | 基于筛选后的论文生成中文回答或 PaperRadar 报告。 |
| Hallucination Checker | 对答案做 groundedness 检查，降低脱离来源文档的风险。 |
| Synthesizer | 清理引用、整理最终答案，并写入输出消息。 |

## 长期记忆设计

项目的长期记忆不是数据库服务，而是轻量 JSON 文件，方便本地 demo、调试和面试讲解。

| 文件 | 保存内容 |
| --- | --- |
| `backend/data/memory/user_topics.json` | 用户长期关注的研究主题。 |
| `backend/data/memory/saved_papers.json` | 用户收藏或待读的论文。 |
| `backend/data/memory/reading_history.json` | 历史检索、任务类型和 top papers。 |
| `backend/data/memory/chat_sessions.json` | 聊天会话、助手回答、压缩摘要和重要笔记。 |

当聊天消息变多时，系统会保留最近消息，并把更早的内容压缩成 `summary` 和 `important_notes`。这样既不会无限增长上下文，也能让后续生成报告时参考用户的历史偏好。

## Mock 模式和真实模式

### Mock 模式

`LLM_PROVIDER=mock` 是默认模式，适合本地快速演示。

- 不需要 DeepSeek/Qwen/Gemini API key。
- 不需要下载 embedding 模型。
- 不真实调用论文源。
- 输出稳定，适合面试展示和自动化测试。

### 真实模式

`LLM_PROVIDER=deepseek|qwen|gemini` 会走真实模型调用，并配合真实论文检索。

- arXiv/PubMed/OpenAlex 会返回真实论文元数据。
- ChromaDB 会持久化向量检索数据。
- sentence-transformers 会生成 embedding。
- DeepSeek 和 Qwen 使用 OpenAI-compatible API。

## 本地安装

建议环境：

- Python 3.11 或 3.12
- Node.js 18+
- Windows PowerShell 或类 Unix shell

### 1. 克隆项目

```powershell
git clone https://github.com/yangzeha/PaperRadar-Agent.git
cd PaperRadar-Agent
```

### 2. 安装后端

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
copy .env.example .env
```

默认 `.env.example` 使用 mock 模式：

```env
LLM_PROVIDER=mock
```

启动后端：

```powershell
uvicorn app.main:app --reload
```

后端默认运行在：

```text
http://localhost:8000
```

健康检查：

```text
http://localhost:8000/health
```

### 3. 安装前端

打开新的终端：

```powershell
cd frontend
npm install
npm run dev
```

前端默认运行在：

```text
http://localhost:3000
```

如果网络较慢，可以使用 npm 镜像：

```powershell
npm install --registry=https://registry.npmmirror.com
```

## 配置真实 LLM Provider

### DeepSeek

```env
LLM_PROVIDER=deepseek
LLM_MODEL_ID=deepseek-chat
LLM_API_KEY=你的_key
LLM_BASE_URL=https://api.deepseek.com
```

### Qwen/DashScope

```env
LLM_PROVIDER=qwen
LLM_MODEL_ID=qwen-plus
LLM_API_KEY=你的_key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### Gemini

```env
LLM_PROVIDER=gemini
LLM_API_KEY=你的_key
LLM_MODEL_ID=gemini-2.5-flash
```

真实 RAG 依赖安装：

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[rag]"
python -m pip install -e ".[providers]"
```

注意：真实 `.env` 不要提交到 GitHub。

## 日常使用

1. 启动后端：

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

2. 启动前端：

```powershell
cd frontend
npm run dev
```

3. 打开页面：

```text
http://localhost:3000
```

4. 输入研究方向：

```text
LLM Agent 长期记忆机制论文雷达
```

5. 查看结果：

- `Research Summary` 展示生成的中文报告。
- `Sources` 展示引用到的论文。
- `Thinking Steps` 展示 Agent 每个节点的执行情况。
- 左侧会话栏保存历史聊天。
- 长期记忆会记录用户主题、历史检索和会话摘要。

## 常用演示问题

```text
Agentic RAG 方向论文雷达：趋势、代表论文、研究空白和两周阅读路线
```

```text
LLM Agent 长期记忆机制论文雷达
```

```text
RAG Hallucination Evaluation 的研究趋势、代表论文和小项目建议
```

```text
Graph Contrastive Learning 推荐系统方向论文雷达
```

```text
帮我找近三年 Multi-Agent RAG 相关论文，并按方法路线分类
```

## API 简表

| 接口 | 方法 | 用途 |
| --- | --- | --- |
| `/health` | GET | 健康检查 |
| `/api/provider` | GET | 查看当前 LLM Provider 状态 |
| `/api/search` | POST | 运行完整 Agent 检索和生成流程 |
| `/api/translate` | POST | 翻译生成报告，并保留 Markdown 和引用 |
| `/api/memory/topics` | GET/POST | 读取或更新长期关注主题 |
| `/api/memory/saved-papers` | GET/POST | 读取或保存待读论文 |
| `/api/memory/history` | GET | 读取历史检索记录 |
| `/api/chat/sessions` | GET/POST | 读取或创建聊天会话 |
| `/api/chat/sessions/{session_id}` | GET/DELETE | 读取或删除指定会话 |
| `/ws/search` | WebSocket | 流式返回 Agent 执行步骤和最终结果 |

## LangGraph Studio

项目根目录提供 `langgraph.json`，可以用 LangGraph Studio 查看图结构、节点输入输出和 state 变化。

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[studio]"
cd ..
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
.\backend\.venv\Scripts\langgraph.exe dev --allow-blocking --no-browser --port 2024 --config langgraph.json
```

打开：

```text
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

## Docker 使用

```powershell
docker compose up --build
```

服务地址：

```text
前端：http://localhost:3000
后端：http://localhost:8000
```

## 测试与验证

后端测试：

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
pytest
```

mock 模式烟测：

```powershell
python scripts/smoke_paper_radar.py
```

报告质量烟测：

```powershell
python scripts/smoke_report_quality.py
```

Provider 检查：

```powershell
python scripts/smoke_provider.py
```

真实检索烟测：

```powershell
python scripts/smoke_real_retrieval.py
```

前端构建：

```powershell
cd frontend
npm run build
```

## 目录结构

```text
backend/app/main.py                 FastAPI 入口和 API 路由
backend/app/agents/graph.py         LangGraph 流程编排
backend/app/agents/state.py         AgentState 状态定义
backend/app/agents/nodes/           Agent 节点实现
backend/app/services/               LLM、检索、记忆、向量库等服务
backend/app/models/schemas.py       请求和响应数据模型
backend/tests/                      后端测试
frontend/app/page.tsx               前端主页面
frontend/lib/api.ts                 前端 API 调用
frontend/components/                前端组件
docs/PAPER_RADAR.md                 项目讲法和 PaperRadar 说明
remark.md                           面向使用者和面试官的项目说明
```

## 面试讲法

可以用下面这段作为项目介绍：

```text
PaperRadar-Agent 是一个基于 LangGraph 的中文论文雷达与选题追踪 Agent。我把普通 RAG 拆成 Router、Retriever、Grader、Rewriter、Generator、Hallucination Checker 和 Synthesizer 等节点，让论文检索、相关性过滤、查询改写、结构化生成、幻觉检查和引用整理都可观测、可调试。

项目支持 mock/real 两种模式，mock 模式方便本地演示和测试，真实模式可以接入 DeepSeek、Qwen 或 Gemini，并检索 arXiv、PubMed、OpenAlex 等论文源。系统还实现了 JSON 长期记忆，用来保存用户关注主题、待读论文、历史检索和聊天摘要。
```

如果面试官追问技术难点，可以重点讲：

- 为什么用 LangGraph：相比单链式 RAG，图结构更适合表达分支、重试、查询改写和质量检查。
- 如何降低幻觉：生成后用 hallucination score 检查答案是否基于来源论文，分数过高会触发重新生成。
- 如何保证引用可追溯：生成阶段要求使用 `[1]`、`[2]`，Synthesizer 再清理无效引用并对齐论文来源。
- 如何支持本地 demo：mock provider 保证无 API key 时也能稳定跑通流程和测试。
- 长期记忆怎么设计：用 JSON 保存主题、论文收藏、历史和会话摘要，轻量、透明、便于面试展示。

## 注意事项

- `.env`、API key、`.venv/`、`node_modules/`、`data/`、`backend/data/`、日志文件不要提交。
- mock 模式适合演示流程，不代表真实论文检索结果。
- 真实模式会受到 API key、网络和论文源限流影响。
- arXiv 公开端点偶尔会出现 429/503，项目会尽量使用 OpenAlex 作为真实元数据兜底。
