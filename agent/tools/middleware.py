from typing import Callable
from utils.prompt_loader import  load_system_prompts, load_report_prompts
from langchain.agents import AgentState
from langgraph.runtime import Runtime

from utils.logger_handler import logger
from langchain.agents.middleware import wrap_tool_call, before_model, ModelRequest, dynamic_prompt
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command


@wrap_tool_call
def monitor_tool( # 监控工具
    request: ToolCallRequest, # 请求的数据封装
    handler:Callable[[ToolCallRequest], ToolMessage | Command] # 执行的函数本身
) -> ToolMessage | Command:
    logger.info(f"工具：{request.tool_call['name']}")
    logger.info(f"参数：{request.tool_call['args']}")
    try:
        result = handler(request)
        logger.info(f"工具：{request.tool_call['name']}调用成功")
        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True # 在运行环境打上标记：当前任务开启报表生成模式
        return result
    except Exception as e:
        logger.error(f"工具：{request.tool_call['name']}调用失败", {str(e)})
        raise e

@before_model
def log_before_model(
        state: AgentState, # agent的状态记录
        runtime: Runtime # 执行过程中的上下文消息
):
    logger.info(f"模型即将发送消息{len(state['messages'])}条")
    logger.debug(f"{type(state['messages'][-1]).__name__} | {state['messages'][-1].content}")
    return None

@dynamic_prompt # 生成模版前调用函数
def report_prompt_switch(request: ModelRequest): # 转换提示词模版
    is_report = request.runtime.context.get("report", False) # 捕捉report标记，默认值false
    if is_report:
        return load_report_prompts()
    return load_system_prompts()

