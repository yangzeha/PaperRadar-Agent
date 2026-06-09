# PaperRadar-Agent

基于 ScholarAgent 二次开发的中文论文雷达与选题追踪 Agent。项目保留 LangGraph Agentic RAG 图流程，并扩展中文 PaperRadar 报告、长期记忆、国内模型 Provider。

`LLM_PROVIDER=deepseek|qwen|gemini` ：

- 真实调用 arXiv/PubMed。
- 使用 ChromaDB 持久化向量库。
- 使用 sentence-transformers 生成 embedding。
- 使用配置的 LLM Provider 生成 PaperRadar 报告。
- 适合最终简历 Demo 和真实验收。

说明：arXiv 官方公开端点偶尔会返回 429/503。真实 smoke 不会退回 mock；如果官方端点被限流，会使用 OpenAlex 真实论文元数据兜底，并在输出中标记 `arxiv_status=ok_openalex_fallback`。

## 功能

- 中文论文雷达：方向概览、方法路线分类、代表论文推荐、近年趋势、研究空白、两周阅读路线、可做小项目建议、参考来源。
- LangGraph 流程：Router、Retriever、Grader、Rewriter、Generator、Hallucination Checker、Synthesizer。
- 长短期记忆：短期状态在 LangGraph state，长期记忆写入 `backend/data/memory/*.json`。
- 记忆 API：topics、saved papers、history。
- 国内模型：DeepSeek 和 Qwen/DashScope 走 OpenAI-compatible API。
- 前端状态提示：页面会显示当前为 mock 演示模式，或显示真实 provider 名称。
- ![Uploading image.png…]()


## PaperRadar 报告质量标准

PaperRadar 不是简单论文列表，而是“方向雷达”。合格报告必须：

- 固定输出 8 个章节：方向概览、方法路线分类、代表论文推荐、近年趋势、研究空白、两周阅读路线、可做小项目建议、参考来源。
- “方法路线分类”必须是路线表，覆盖 Survey / Taxonomy、Planning / Reasoning、Multi-Agent / Hierarchical、Multimodal、Evaluation / Benchmark、Domain-specific 等实际检索到的路线。
- “代表论文推荐”必须解释为什么代表该方向，而不是截断摘要。
- “近年趋势”要按年份或阶段归纳，并绑定引用。
- “研究空白”至少给出 5 个具体 gap，每个 gap 包含现状、缺口、可验证方式和两周 demo。
- “两周阅读路线”必须按学习目标安排，不按论文编号机械排序。
- “可做小项目建议”必须贴合 Agentic RAG、LangGraph、RAG、多 Agent、长短期记忆、citation grounding 或 query rewrite。
- “参考来源”必须尽量包含 title、authors、year/published date、source、url。
- mock 模式和 real 模式都必须遵守同一报告结构；mock 只影响数据来源，不降低报告结构要求。

## Windows 安装

建议先使用 Python 3.11 或 3.12。真实 RAG 依赖比较重，尤其是 ChromaDB、sentence-transformers 和 PyTorch 相关下载，第一次安装可能较慢。

### 1. 后端基础依赖，支持 mock Demo

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
copy .env.example .env
```

启动：

```powershell
uvicorn app.main:app --reload
```

### 2. 测试依赖

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
pytest
```

### 3. 真实 RAG 依赖

如果 `pip install -e ".[dev]"` 超时，不要一次性装全量，按下面拆开：

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[rag]"
python -m pip install -e ".[providers]"
```

最小真实检索依赖也可以手动安装：

```powershell
python -m pip install chromadb sentence-transformers arxiv langchain-chroma langchain-community httpx
```

如果国内网络慢，可以加镜像：

```powershell
python -m pip install -e ".[rag]" -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -e ".[providers]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

依赖组说明：

- 基础依赖：FastAPI、LangGraph、httpx、pydantic-settings，用于 mock Demo。
- `.[rag]`：arXiv、ChromaDB、langchain-chroma、sentence-transformers，用于真实检索。
- `.[providers]`：Gemini 的 LangChain Provider；DeepSeek/Qwen 只依赖基础 httpx。
- `.[test]`：pytest、pytest-asyncio、respx、ruff。
- `.[dev]`：全量开发依赖，不建议网络慢时直接安装。

## 前端启动

```powershell
cd frontend
npm install --registry=https://registry.npmmirror.com
npm run dev
```

打开 `http://localhost:3000`。

## LangGraph Studio 可视化

项目根目录已经提供 `langgraph.json`，会把 `paper_radar` 图指向
`backend/app/agents/graph.py:graph`。启动后可以在 Studio 里查看节点、边、
运行状态和每一步的 state。

先安装 Studio CLI：

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[studio]"
```

从项目根目录启动 LangGraph Server：

```powershell
cd ..
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
.\backend\.venv\Scripts\langgraph.exe dev --allow-blocking --no-browser --port 2024 --config langgraph.json
```

打开：

- Studio UI：`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`
- API Docs：`http://127.0.0.1:2024/docs`
- 健康检查：`http://127.0.0.1:2024/ok`

说明：Windows 终端默认 GBK 编码可能导致 LangGraph CLI 读取 Unicode 文本时报错，
所以建议设置 `PYTHONUTF8=1` 和 `PYTHONIOENCODING=utf-8`。
如果只是演示，不需要热更新，可以在命令末尾加 `--no-reload`，避免日志文件或缓存变更触发反复重载。
前端右上角的 `Studio` 按钮也会跳到这个 Studio UI。

## 模型配置

Mock：

```env
LLM_PROVIDER=mock
```

DeepSeek：

```env
LLM_PROVIDER=deepseek
LLM_MODEL_ID=deepseek-chat
LLM_API_KEY=你的_key
LLM_BASE_URL=https://api.deepseek.com
```

DeepSeek 同时兼容旧变量 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`，但通用变量优先。`LLM_BASE_URL` 可以填 `https://api.deepseek.com`、`https://api.deepseek.com/v1` 或完整 `/chat/completions`，代码会自动补齐。

Qwen/DashScope：

```env
LLM_PROVIDER=qwen
LLM_MODEL_ID=qwen-plus
LLM_API_KEY=你的_key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

Qwen 同时兼容旧变量 `DASHSCOPE_API_KEY`、`QWEN_BASE_URL`、`QWEN_MODEL`，但通用变量优先。`LLM_BASE_URL` 可以填 `https://dashscope.aliyuncs.com`、`https://dashscope.aliyuncs.com/compatible-mode/v1` 或完整 `/chat/completions`，代码会自动补齐。

Gemini：

```env
LLM_PROVIDER=gemini
LLM_API_KEY=你的_key
LLM_MODEL_ID=gemini-2.5-flash
```

## 验证命令

后端测试：

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
pytest
```

Mock PaperRadar：

```powershell
python scripts/smoke_paper_radar.py
```

PaperRadar 报告质量：

```powershell
python scripts/smoke_report_quality.py
```

Provider：

```powershell
python scripts/smoke_provider.py
```

真实检索：

```powershell
$env:LLM_PROVIDER="deepseek"
$env:LLM_MODEL_ID="deepseek-chat"
$env:LLM_API_KEY="你的_key"
$env:LLM_BASE_URL="https://api.deepseek.com"
python scripts/smoke_real_retrieval.py
```

如果没有 API key，`smoke_provider.py` 和 `smoke_real_retrieval.py` 会输出 `SKIP`，不会失败。

前端构建：

```powershell
cd frontend
npm run build
```

## 记忆 API

- `GET /api/memory/topics`
- `POST /api/memory/topics`
- `GET /api/memory/saved-papers`
- `POST /api/memory/saved-papers`
- `GET /api/memory/history`

JSON 文件不存在时会自动创建；JSON 损坏时会 graceful fallback 成空结构，不会让服务崩溃。

## 示例问题

- `Agentic RAG 方向论文雷达：趋势、代表论文、研究空白和两周阅读路线`
- `LLM Agent 长期记忆机制论文雷达`
- `RAG Hallucination Evaluation 的研究趋势、代表论文和小项目建议`

更多项目讲法见 [docs/PAPER_RADAR.md](docs/PAPER_RADAR.md)。
