from typing import List

from dotenv import load_dotenv
from langchain_core.tools import tool, BaseTool
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolSelectorMiddleware, AgentMiddleware
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from llm import llm3,llm

load_dotenv()


@tool
def tool_1(input:str) -> str:
    """
    This is a useless tool, intended solely as an example.
    """
    return "This is a useless tool, intended solely as an example."
@tool
def tool_2(input:str) -> str:
    """
    This is a useless tool, intended solely as an example.
    """
    return "This is a useless tool, intended solely as an example."
@tool
def tool_3(input:str) -> str:
    """
    This is a useless tool, intended solely as an example.
    """
    return "This is a useless tool, intended solely as an example."
@tool
def tool_4(input:str) -> str:
    """
    This is a useless tool, intended solely as an example.
    """
    return "This is a useless tool, intended solely as an example."


@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations and return the result.
    Args:
        expression: Mathematical expression to evaluate
        (e.g., "2 + 3 * 4", "sqrt(16)", "sin(pi/2)")
    Returns:
        The calculated result as a string
    """
    result = str(eval(expression))
    return result


class DeepSeekCompatibleToolSelector(AgentMiddleware):
    """为 DeepSeek 定制的工具筛选中间件，使用普通 API 而非结构化输出"""

    def __init__(self, model: ChatOpenAI, max_tools: int = 10):
        self.model = model
        self.max_tools = max_tools

    def wrap_model_call(self, request, handler):
        """在模型调用前筛选工具"""
        # 获取用户的最新消息
        last_message = request.messages[-1]
        user_query = last_message.content if hasattr(last_message, 'content') else str(last_message)

        # 使用普通聊天 API（非结构化输出）让模型选择工具
        selection_prompt = f"""
        用户问题：{user_query}

        可用工具列表：
        {self._format_tools(request.tools)}

        请选择与用户问题最相关的 {self.max_tools} 个工具。
        只返回工具名称，用逗号分隔，例如：search_web, calculate

        注意：只返回工具名称，不要有其他内容。
        """

        # 调用模型获取工具选择结果（使用普通 API）
        response = self.model.invoke(selection_prompt)
        selected_names = [name.strip() for name in response.content.split(",")]

        # 筛选工具
        filtered_tools = [
            tool for tool in request.tools
            if tool.name in selected_names
        ]

        # 确保 always_include 工具始终存在
        always_include = getattr(self, 'always_include', [])
        for tool_name in always_include:
            for tool in request.tools:
                if tool.name == tool_name and tool not in filtered_tools:
                    filtered_tools.append(tool)

        # 更新请求中的工具列表
        request.tools = filtered_tools

        # 继续执行
        return handler(request)

    def _format_tools(self, tools: List[BaseTool]) -> str:
        """格式化工具列表供模型选择"""
        lines = []
        for tool in tools:
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)



agent = create_deep_agent(
    model=llm3,
    tools=[tool_1, tool_2, tool_3, tool_4,calculate],
    # middleware=[
    #     DeepSeekCompatibleToolSelector(
    #         model=llm3,
    #         max_tools=2,
    #     ),
    # ],
)

status = {
    "messages": [{"role": "user", "content": "请计算2+3*4的值"}]
}

result = agent.invoke(status)
print(result)