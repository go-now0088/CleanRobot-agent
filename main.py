"""
扫地机器人智能客服 —— FastAPI 服务入口

把 agent.react_agent.ReactAgent 包装成 HTTP 服务，供前后端分离调用：

    POST   /chat/stream           流式对话，SSE 返回（前端推荐）
    POST   /chat                  一次性对话，直接返回完整答案
    GET    /health                健康检查
    GET    /api/sessions          会话列表
    GET    /api/sessions/{id}     查询某个会话的历史消息
    DELETE /api/sessions/{id}     清空某个会话

启动：
    python main.py
    # 或（支持热重载）
    uvicorn main:app --host 127.0.0.1 --port 8000 --reload

接口文档：
    Swagger  http://127.0.0.1:8000/docs
    ReDoc    http://127.0.0.1:8000/redoc
    OpenAPI  http://127.0.0.1:8000/openapi.json
"""
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Iterator, Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from agent.react_agent import ReactAgent
from utils.config_handler import rag_config
from utils.logger_handler import logger

# 发给模型的历史消息条数上限，避免多轮对话越聊越贵（保留最近 N 条）
MAX_HISTORY_MESSAGES = 20


# ---------------------------------------------------------------------------
# Agent 与 会话存储
# ---------------------------------------------------------------------------
_agent: Optional[ReactAgent] = None

# session_id -> [{"role": "user"/"assistant", "content": str, "time": float}, ...]
_sessions: dict[str, list[dict]] = {}
_sessions_lock = threading.Lock()   # 线程锁，让他同一时间只有一个再跑


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时初始化 Agent（加载模型、提示词、向量库），避免首个请求等太久"""
    global _agent
    logger.info("正在初始化 ReactAgent（加载模型 / 提示词 / 向量库）...")
    start = time.time()
    _agent = ReactAgent()
    logger.info(f"ReactAgent 初始化完成，耗时 {time.time() - start:.2f}s，服务已就绪")
    yield
    _agent = None
    logger.info("服务已关闭")

app = FastAPI(
    title="扫地机器人智能客服 API",
    description=(
        "基于 LangChain Agent + RAG 的扫地机器人智能客服，支持多轮对话、工具调用与报表生成。\n\n"
        "- 会话由 `session_id` 标识：**不传**则服务端新建并在 `start` 事件 / 响应体里返回，"
        "后续请求带上同一个 `session_id` 即为多轮对话。\n"
        "- 会话历史保存在服务进程内存中，**重启即丢失**，且多 worker 部署时各 worker 独立。"
    ),
    version="1.0.0",
    openapi_tags=[
        {"name": "对话", "description": "问答接口，流式与非流式两个版本"},
        {"name": "会话", "description": "会话历史的查询与清空"},
        {"name": "运维", "description": "健康检查等"},
    ],
    lifespan=lifespan,
)

# 前后端分离部署时浏览器需要跨域访问，这里放开所有来源；生产环境请改成具体域名
app.add_middleware(
    CORSMiddleware,     # 跨域资源共享
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_agent() -> ReactAgent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent 尚未就绪，请稍后重试")
    return _agent


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., description="用户提问")
    session_id: Optional[str] = Field(
        None, max_length=64, description="会话 ID，不传由服务端新建；传入同一 ID 即可多轮对话"
    )
    report: bool = Field(False, description="是否直接开启报表生成模式")

    @field_validator("query")
    @classmethod
    def _check_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query 不能为空")
        return value

    @field_validator("session_id")
    @classmethod
    def _check_session_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


class ChatResponse(BaseModel):
    session_id: str = Field(..., description="会话 ID，下次请求带上它即可继续多轮对话")
    answer: str = Field(..., description="完整回答")
    elapsed: float = Field(..., description="服务端耗时（秒）")


class HealthResponse(BaseModel):
    status: Literal["ok", "initializing"] = Field(..., description="ok=就绪；initializing=Agent 还在初始化")
    chat_model: str = Field(..., description="当前使用的对话模型名")
    sessions: int = Field(..., description="内存中的会话数")


class SessionInfo(BaseModel):
    session_id: str
    messages: int = Field(..., description="该会话累计的消息条数")
    updated_at: Optional[float] = Field(None, description="最后一条消息的时间戳（秒，Unix epoch）")


class SessionListResponse(BaseModel):
    total: int
    sessions: list[SessionInfo]


class SessionMessage(BaseModel):
    role: Literal["user", "assistant"]   # 限制输出
    content: str
    time: float = Field(..., description="写入时间戳（秒，Unix epoch）")


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: list[SessionMessage]


class ClearSessionResponse(BaseModel):
    session_id: str
    cleared: bool = Field(..., description="是否清空成功（会话不存在时返回 404，不会出现在这里）")


# ---------------------------------------------------------------------------
# 会话存储辅助函数
# ---------------------------------------------------------------------------
def _normalize_session_id(session_id: Optional[str]) -> str:
    """没带 session_id 就新建一个（方便前端首次调用后记住）"""
    if session_id:
        return session_id
    return uuid.uuid4().hex     # 自动编号，转换成32位16进制


def _get_history(session_id: str) -> list[dict]:
    """取出历史消息（只保留 role/content，并截断到最近 MAX_HISTORY_MESSAGES 条）"""
    with _sessions_lock:     # 自动开锁关锁
        messages = _sessions.get(session_id, [])    # id存在，返回消息，不存在返回[]
        recent = messages[-MAX_HISTORY_MESSAGES:]     # 切片，拿到后几条消息
    return [{"role": m["role"], "content": m["content"]} for m in recent]    # 列表推导式


def _append_message(session_id: str, role: str, content: str) -> None:
    with _sessions_lock:
        _sessions.setdefault(session_id, []).append(     # 字典写法，如果id存在就返回key，不存在就创建，值是[]
            {"role": role, "content": content, "time": time.time()}
        )


# ---------------------------------------------------------------------------
# SSE 工具
# ---------------------------------------------------------------------------
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _iter_chat_events(session_id: str, query: str, is_report: bool) -> Iterator[str]:
    """
    迭代对话事件
    真正干活的地方：调用 Agent 流式生成，并把结果转成 SSE 事件流。

    Agent 每次产出的是「当前最新消息的完整内容」快照，所以这里做前缀比较算出增量，
    前端拿到 content 直接往气泡后面追加即可。

    事件约定：
        event: start   data: {"session_id": "..."}
        event: message data: {"content": "本次新增的文本"}
        event: done    data: {"session_id": "...", "answer": "完整答案", "elapsed": 1.23}
        event: error   data: {"message": "错误信息"}
    """
    history = _get_history(session_id)
    _append_message(session_id, "user", query)
    yield _sse("start", {"session_id": session_id})

    answer = ""
    previous: Optional[str] = None
    start = time.time()
    try:
        for snapshot in get_agent().execute_stream(query, is_report=is_report, history=history):
            if not snapshot.startswith(previous or ""):  # 如果新快照不是在上一次快照基础上追加，上一条消息结束、新的一条开始
                previous = None
            # 把之前已经发给前端的消息全部跳过，只拿后面新增的消息，之前的快照不为空取增量，反之返回整条消息
            delta = snapshot[len(previous):] if previous else snapshot
            previous = snapshot
            if not delta:
                continue
            answer += delta
            yield _sse("message", {"content": delta})
    except Exception as e:  # Agent 出错时把错误推给前端，而不是直接断流
        logger.error(f"[chat] 会话 {session_id} 生成失败：{str(e)}", exc_info=True)
        yield _sse("error", {"message": f"生成失败：{str(e)}"})
        return

    answer = answer.strip()
    if answer:
        _append_message(session_id, "assistant", answer)
    logger.info(f"[chat] 会话 {session_id} 完成，耗时 {time.time() - start:.2f}s，回答 {len(answer)} 字")
    yield _sse("done", {
        "session_id": session_id,
        "answer": answer,
        "elapsed": round(time.time() - start, 2),
    })


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["运维"], summary="健康检查")
def health():
    """探活用。Agent 未就绪时也能访问（返回 initializing），可用它做启动后的就绪探测。"""
    return HealthResponse(
        status="ok" if _agent is not None else "initializing",
        chat_model=rag_config["chat_model_name"],
        sessions=len(_sessions),
    )


@app.post(
    "/chat/stream",
    tags=["对话"],
    summary="流式对话（SSE）",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "SSE 事件流，事件类型见下方说明",
            "content": {
                "text/event-stream": {
                    "example": (
                        'event: start\n'
                        'data: {"session_id": "7c9e6679b1f04f0e8b3d2a5c1e6f8a90"}\n\n'
                        'event: message\n'
                        'data: {"content": "滤网建议每"}\n\n'
                        'event: message\n'
                        'data: {"content": "1-2 周清洗一次。"}\n\n'
                        'event: done\n'
                        'data: {"session_id": "7c9e6679b1f04f0e8b3d2a5c1e6f8a90", '
                        '"answer": "滤网建议每1-2 周清洗一次。", "elapsed": 3.21}\n\n'
                    )
                }
            },
        },
        503: {"description": "Agent 尚未就绪，请稍后重试"},
    },
)
def chat_stream(req: ChatRequest):
    """
    POST 提交问题，响应体为 `text/event-stream`，前端按 SSE 协议逐事件读取。

    事件约定（每个事件以空行分隔，`data` 均为 JSON）：

    | event | data 字段 | 说明 |
    | --- | --- | --- |
    | `start` | `session_id` | 流开始。请求没带 `session_id` 时，服务端在这里下发新建的 ID，前端需记住它 |
    | `message` | `content` | 本次新增的一段文本，**追加**到气泡后面即可（是增量，不要整段覆盖） |
    | `done` | `session_id` / `answer` / `elapsed` | 本次回答结束，`answer` 是完整答案 |
    | `error` | `message` | 生成失败，流随即结束（HTTP 状态码此时仍是 200，需靠事件判断） |

    两点注意：

    1. 只有 `start` 之后的 `message` 才是模型回答，**不会**出现用户提问的回显或工具检索的原始资料。
    2. 当前服务端的流式粒度是「整条消息」（LangChain `values` 模式每个节点执行完才吐快照），
       不是逐 token，所以通常一次就收到整段文本；要真正的逐 token 打字机效果需要改
       `ReactAgent.execute_stream` 的 `stream_mode`。
    """
    get_agent()  # 提前校验，未就绪时直接返回 503 而不是流里报错
    session_id = _normalize_session_id(req.session_id)
    return StreamingResponse(
        # 生成器是同步阻塞的（LangChain 调用），Starlette 会自动放到线程池里执行，不阻塞事件循环
        _iter_chat_events(session_id, req.query, req.report),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",    # 禁止浏览器 / 代理缓存 SSE 流，实时获取每一条事件。
            "Connection": "keep-alive",    # 保持长连接，持续接收流式数据。
            "X-Accel-Buffering": "no",  # 让 Nginx 之类不要缓冲，保证实时性
        },
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["对话"],
    summary="一次性对话（不流式）",
    responses={
        500: {"description": "生成失败"},
        503: {"description": "Agent 尚未就绪，请稍后重试"},
    },
)
def chat(req: ChatRequest):
    """不想处理 SSE 时用这个：服务端把整段答案算完再一次性返回。行为和 `/chat/stream` 一致，同样写入会话历史。"""
    agent = get_agent()
    session_id = _normalize_session_id(req.session_id)
    history = _get_history(session_id)
    _append_message(session_id, "user", req.query)

    start = time.time()
    answer = ""
    try:
        for snapshot in agent.execute_stream(req.query, is_report=req.report, history=history):
            answer = snapshot  # 快照是累积内容，最后一次就是完整答案
    except Exception as e:
        logger.error(f"[chat] 会话 {session_id} 生成失败：{str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成失败：{str(e)}")

    answer = answer.strip()
    if answer:
        _append_message(session_id, "assistant", answer)
    return ChatResponse(session_id=session_id, answer=answer, elapsed=round(time.time() - start, 2))


@app.get("/api/sessions", response_model=SessionListResponse, tags=["会话"], summary="会话列表")
def list_sessions():
    """列出当前进程内存里的所有会话（重启后为空，多 worker 部署时只反映处理该请求的那个 worker）。"""
    with _sessions_lock:
        return SessionListResponse(
            total=len(_sessions),
            sessions=[
                SessionInfo(
                    session_id=sid,
                    messages=len(messages),
                    updated_at=messages[-1]["time"] if messages else None,
                )
                for sid, messages in _sessions.items()
            ],
        )


@app.get(
    "/api/sessions/{session_id}",
    response_model=SessionHistoryResponse,
    tags=["会话"],
    summary="查询会话历史",
    responses={404: {"description": "会话不存在"}},
)
def get_session(session_id: str = PathParam(..., max_length=64, description="会话 ID")):
    """取出往来的完整消息（只含 user / assistant，即用户提问与最终回答）。"""
    with _sessions_lock:
        messages = _sessions.get(session_id)    # 存在返回对应的消息列表，反之返回none
    if messages is None:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
    return SessionHistoryResponse(session_id=session_id, messages=messages)


@app.delete(
    "/api/sessions/{session_id}",
    response_model=ClearSessionResponse,
    tags=["会话"],
    summary="清空会话",
    responses={404: {"description": "会话不存在"}},
)
def delete_session(session_id: str = PathParam(..., max_length=64, description="会话 ID")):
    """删除该会话的全部历史。下次再带同一个 `session_id` 请求，等价于新会话。"""
    with _sessions_lock:
        # 取出并删除，存在删除value并返回，反之none。判断 pop 返回的结果是不是不等于 None，得到布尔值：存在是true，没有是false
        existed = _sessions.pop(session_id, None) is not None
    if not existed:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
    return ClearSessionResponse(session_id=session_id, cleared=True)


@app.get("/", include_in_schema=False)
def index():
    """
    重定向响应。
    :return: 浏览器访问根地址 `http://127.0.0.1:8000/`后端返回302重定向指令，浏览器自动跳转到 `http://127.0.0.1:8000/docs`
    """
    return RedirectResponse("/docs")


if __name__ == '__main__':
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    logger.info(f"启动服务：http://{host}:{port}   （接口文档 /docs）")
    uvicorn.run(app, host=host, port=port, log_level="info")
