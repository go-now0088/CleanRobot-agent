# 扫地机器人智能客服

基于 **LangChain Agent + RAG** 的扫地机器人 / 扫拖一体机器人智能客服，对外提供 **FastAPI 接口**。

Agent 具备自主的 ReAct 思考与工具调用能力，可检索本地知识库、查询环境与用户数据，并在识别出「生成个人使用报告」意图时自动切换到报表提示词完成报告生成。

## 功能特性

- **多轮对话**：以 `session_id` 标识会话，服务端保留上下文（默认携带最近 20 条历史消息）
- **流式输出**：`/chat/stream` 以 SSE 逐段下发回答，适合前端打字机效果
- **RAG 检索**：TXT / PDF 知识库切分入库（Chroma），MD5 去重，支持增量添加
- **工具调用**：7 个工具，覆盖知识检索、天气查询、用户定位、用户 ID、当前月份、外部使用记录、报表上下文注入
- **报表生成**：识别报告类意图后由中间件动态切换系统提示词（`dynamic_prompt`）
- **可观测性**：工具调用与模型调用均写入日志（控制台 INFO / 文件 DEBUG）

## 技术栈

| 分类 | 选型 |
| --- | --- |
| Web 框架 | FastAPI 0.141 + Uvicorn |
| Agent 框架 | LangChain 1.3（`create_agent`）+ LangGraph |
| 大模型 | 通义千问 `qwen3-max`（`ChatTongyi`） |
| 向量模型 | `text-embedding-v4`（`DashScopeEmbeddings`） |
| 向量库 | Chroma 1.5 |
| 配置 | PyYAML（`config/*.yml`） |

## 目录结构

```
.
├── main.py                  # FastAPI 服务入口（对话 / 会话 / 健康检查）
├── app.py                   # Streamlit 版界面（可选，见「说明与限制」）
├── agent/
│   ├── react_agent.py       # ReactAgent：组装模型、提示词、工具、中间件
│   └── tools/
│       ├── agent_tools.py   # 7 个工具定义
│       └── middleware.py    # 工具监控 / 模型调用日志 / 提示词动态切换
├── rag/
│   ├── vector_store.py      # 向量库封装、文档切分入库（MD5 去重）
│   └── rag_service.py       # 检索 + 总结链
├── model/factory.py         # 模型工厂（对话模型 / 向量模型）
├── utils/                   # 配置、提示词、文件、日志、路径工具
├── config/                  # rag.yml / chroma.yml / agent.yml / prompts.yml
├── prompts/                 # 系统提示词、RAG 提示词、报表提示词
├── data/                    # 知识库（txt / pdf）与外部数据（data/external/records.csv）
└── requirements.txt
```

## 快速开始

### 1. 环境要求

- Python **3.10+**（实测环境：Python 3.14.3 / Windows 11）
- 阿里云百炼（DashScope）API Key：[控制台获取](https://bailian.console.aliyun.com/)

### 2. 安装依赖

```bash
# Windows
.venv\Scripts\python.exe -m pip install -r requirements.txt

# macOS / Linux
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 配置 API Key

模型与向量化都走 DashScope，SDK 会自动读取环境变量 `DASHSCOPE_API_KEY`，**代码里不存密钥**：

```bash
# Windows (PowerShell，当前会话)
$env:DASHSCOPE_API_KEY = "sk-xxxxxxxxxxxxxxxx"

# Windows (cmd，永久)
setx DASHSCOPE_API_KEY "sk-xxxxxxxxxxxxxxxx"

# macOS / Linux
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

### 4. 构建向量库

知识库文件放在 `data/` 下（支持 `.txt` / `.pdf`），执行：

```bash
python rag/vector_store.py
```

脚本会遍历 `data/`、按 MD5 去重、切分后写入 `chroma_db/`（collection 名 `agent`），并打印一条检索示例。文档有更新时重复执行即可增量入库。

### 5. 启动服务

```bash
python main.py

# 或带热重载
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

启动后访问 **<http://127.0.0.1:8000/docs>** 查看 Swagger 文档（根路径 `/` 会自动跳转到 `/docs`）。

| 文档 | 地址 |
| --- | --- |
| Swagger UI | `http://127.0.0.1:8000/docs` |
| ReDoc | `http://127.0.0.1:8000/redoc` |
| OpenAPI JSON | `http://127.0.0.1:8000/openapi.json` |

> 服务启动时会预加载模型、提示词与向量库（`lifespan`），期间 `/health` 返回 `initializing`，可用于就绪探测。

## 接口说明

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/chat/stream` | 流式对话，SSE 返回 |
| POST | `/chat` | 一次性对话，算完再返回 |
| GET | `/health` | 健康检查（含当前对话模型、会话数） |
| GET | `/api/sessions` | 会话列表 |
| GET | `/api/sessions/{session_id}` | 查询会话历史消息 |
| DELETE | `/api/sessions/{session_id}` | 清空会话 |

请求体（`/chat` 与 `/chat/stream` 相同）：

```json
{
  "query": "滤网多久清洗一次？",
  "session_id": null,
  "report": false
}
```

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `query` | string | 是 | 用户提问，不允许为空 |
| `session_id` | string | 否 | 会话 ID；不传由服务端新建，传入同一 ID 即为多轮对话 |
| `report` | bool | 否 | 直接开启报表生成模式（默认 `false`，通常由模型自行判断） |

### 非流式调用

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"滤网多久清洗一次？"}'
```

```json
{
  "session_id": "7c9e6679b1f04f0e8b3d2a5c1e6f8a90",
  "answer": "滤网建议每 1-2 周清洗一次……",
  "elapsed": 3.21
}
```

### 流式调用（SSE）

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"别墅适合用什么机器人打扫？"}'
```

响应为 `text/event-stream`，事件约定：

| event | data 字段 | 说明 |
| --- | --- | --- |
| `start` | `session_id` | 流开始；请求未带 `session_id` 时，服务端在此下发新建的 ID，前端需记住 |
| `message` | `content` | 本次新增的文本，**追加**到气泡后即可（增量，不是整段覆盖） |
| `done` | `session_id` / `answer` / `elapsed` | 回答结束，`answer` 为完整答案 |
| `error` | `message` | 生成失败，流随即结束（此时 HTTP 状态码仍为 200，需按事件判断） |

示例：

```
event: start
data: {"session_id": "7c9e6679b1f04f0e8b3d2a5c1e6f8a90"}

event: message
data: {"content": "滤网建议每"}

event: done
data: {"session_id": "7c9e6679b1f04f0e8b3d2a5c1e6f8a90", "answer": "滤网建议每 1-2 周清洗一次。", "elapsed": 3.21}
```

### 多轮对话

首次请求不传 `session_id`，从 `start` 事件（或响应体）拿到服务端生成的 ID，后续请求带上同一个 ID 即可：

```python
import requests

BASE = "http://127.0.0.1:8000"
session_id = None

for question in ["滤网多久清洗一次？", "那滚刷呢？"]:
    body = {"query": question, "session_id": session_id}
    resp = requests.post(f"{BASE}/chat", json=body).json()
    session_id = resp["session_id"]      # 后续请求续用
    print(resp["answer"])
```

## 配置说明

配置集中在 `config/`，改完重启服务生效。

**`config/rag.yml`** — 模型名

```yaml
chat_model_name: qwen3-max
embedding_model_name: text-embedding-v4
```

**`config/chroma.yml`** — 向量库与切分参数

```yaml
collection_name: agent
persist_directory: chroma_db
k: 3                                    # 检索返回条数
data_path: data                         # 知识库目录
md5_hex_store: md5.text                 # MD5 去重记录
allow_knowledge_file_type: ["txt", "pdf"]
chunk_size: 200
chunk_overlap: 20
```

**`config/agent.yml`** — `external_data_path: data/external/records.csv`（工具 `fetch_external_data` 的数据源）

**`config/prompts.yml`** — 三个提示词模板的路径，模板文件在 `prompts/` 下

### 服务端口

通过环境变量覆盖（默认 `127.0.0.1:8000`）：

```bash
API_HOST=0.0.0.0 API_PORT=9000 python main.py
```

## 单独调试各模块

```bash
python agent/react_agent.py      # 命令行跑一次完整 Agent 问答
python rag/rag_service.py        # 只测检索 + 总结
python rag/vector_store.py       # 重建 / 增量更新向量库
```

## 说明与限制

- **会话存在进程内存中**：服务重启即丢失；多 worker 部署时各 worker 会话相互独立。生产环境建议换成 Redis 之类的共享存储（改 `main.py` 里的 `_sessions`）。
- **流式粒度是「整条消息」**：`ReactAgent.execute_stream` 使用 LangGraph 的 `values` 模式，每个节点执行完才产出快照，所以前端通常一次收到整段文本。要真正的逐 token 打字机效果，需要改用 `messages` 模式的 `stream_mode`。
- **CORS 当前放开所有来源**（`allow_origins=["*"]`），生产部署请改成具体域名。
- **`app.py` 是早期 Streamlit 版本界面**，当前依赖未安装 streamlit，跑不起来；接口（`main.py`）与它互不影响，保留仅作参考。
- **工具数据为演示用途**：天气、用户定位、用户 ID、当前月份均为 mock 数据，接入真实系统时替换 `agent/tools/agent_tools.py` 中的实现即可。
