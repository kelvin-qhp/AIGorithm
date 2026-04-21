# from langchain.tools import tool
# from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

import langchain
# from langgraph.runtime import ExecutionInfo

# @tool
# def search(query: str) -> str:
#     """Search for information."""
#     return f"Results for: {query}"
#
# @tool
# def get_weather(location: str) -> str:
#     """Get weather information for a location."""
#     return f"Weather in {location}: Sunny, 72°F"

if __name__ == '__main__':


    load_dotenv()
    # model = init_chat_model(model="qwen-plus",
    #                         model_provider="openai",
    #                         openai_api_key=os.getenv("OPENAI_API_KEY"),
    #                         openai_api_base=os.getenv("OPENAI_API_BASE"))
    # result = model.invoke([{"role": "system", "content": "You are a helpful assistant that can use tools to answer questions."},
    #               {"role": "user", "content": "What is the weather in shenzhen?"}])
    # print(result)
    model = ChatOpenAI(model_name="qwen-plus",
               openai_api_key=os.getenv("OPENAI_API_KEY"),
               openai_api_base=os.getenv("OPENAI_API_BASE"),
               temperature=0,
               streaming=True)
    response = model.invoke("请列出1个深圳最著名的旅游景点")
    print(response.content)

    # prompt_template= """You are a helpful assistant that can use tools to answer questions."""
    # agent = create_openai_tools_agent(model, tools=[search, get_weather],prompt=prompt_template)
    # response = agent.run("What is the weather in New York?")

