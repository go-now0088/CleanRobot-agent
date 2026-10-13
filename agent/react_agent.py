from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import  monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model = chat_model,
            system_prompt = load_system_prompts(),
            tools = [rag_summarize, get_weather, get_user_location, get_user_id,
                    get_current_month, fetch_external_data, fill_context_for_report],
            middleware = [monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str, is_report: bool = False, history: list[dict] | None = None):
        """
        流式执行一次问答

        :param query: 本次用户提问
        :param is_report: 是否直接开启报表生成模式
        :param history: 历史对话消息 [{"role": "user"/"assistant", "content": str}, ...]，
                        传入后 agent 可感知前文，实现多轮对话；不传则与原来一样是单轮独立问答
        :return: 迭代器，每次产出的是「当前模型回答的完整内容」（快照，不是增量）。
                 只产出模型自己的回答：用户提问的回显和工具返回结果都会被跳过。
        """
        messages = []
        for message in history or []:
            role = message.get("role")
            content = message.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": query})

        input_text = {
            "messages": messages,
        }
        for chunk in self.agent.stream(input_text, stream_mode= "values", context={"report": is_report}):
            latest_message = chunk["messages"][-1]
            # values 模式每次给出的是「最新一条消息」：可能是我们塞进去的 HumanMessage，
            # 也可能是工具调用的 ToolMessage，这些都不该当成回答推给前端，只取 AI 的输出
            if not isinstance(latest_message, AIMessage):
                continue
            content = latest_message.content
            if isinstance(content, str) and content.strip():
                yield content.strip() + "\n"




if __name__ == '__main__':
    agent = ReactAgent()
    for chunk in agent.execute_stream("扫地机器人在我所在的地区的温度下应该怎么保养"):
        print(chunk, end = "", flush=True)