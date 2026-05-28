from dotenv import load_dotenv
from isapi.samples.redirector_with_filter import virtualdir
from langchain.agents import create_agent
# from langchain_deepseek import ChatDeepSeek
from deepagents import FilesystemMiddleware,create_deep_agent
from deepagents.backends import FilesystemBackend, StateBackend, StoreBackend, CompositeBackend
from langchain.messages import HumanMessage
from langgraph.store.memory import InMemoryStore
from llm import llm
load_dotenv()

# model = ChatDeepSeek(
#     model="deepseek-chat",
# )

model = llm

store = InMemoryStore()

composite_backend = CompositeBackend(
    default=StateBackend(),
    routes={
        "/memories/user/": StoreBackend(store=store),
        "/memories/skill/": StoreBackend(store=store,namespace=lambda ctx: getattr(ctx, "thread_id", "default")),
        "/memories/": StoreBackend(store=store)
    }
)

system_prompt="""你是一个具备持久记忆的智能助手。
    你的文件系统能力：
    - 可以读写临时文件（如 /draft.txt），这些文件只在当前会话有效
    - 可以读写持久记忆（路径以 /memories/ 开头），这些记忆会跨会话保存
    
    使用建议：
    - 保存用户偏好时，使用 /memories/user/preferences.txt
    - 保存长期知识时，使用 /memories/knowledge/
    - 临时草稿和中间结果，直接使用根目录
    - 技能定义放在 /memories/skills/ 目录下
    
    请合理使用不同存储区域来管理信息。"""

agent1 = create_deep_agent(
    model=model,
    store=store,
    backend=composite_backend,
    system_prompt=system_prompt
)


# agent1 = create_agent(
#     model=model,
#     store=store,
#     # backend=composite_backend
#     middleware=[
#         FilesystemMiddleware(
#             backend=composite_backend
#         )
#     ]
# )


config1 = {"configurable": {"thread_id": '1'}}

print("\n[线程 A] 写入文件...")
result_a = agent1.invoke(
    {
        "messages": [{
            "role": "user",
            "content": """
            请执行以下操作：
            1. 写入临时文件 /draft.txt，内容："这是临时草稿，仅当前会话可见"
            2. 写入持久记忆 /memories/user/preferences.txt，内容："用户偏好：简洁回答，使用中文"
            """
        }]
    },
    config=config1
)
print(f"[线程 A] 执行完成")



# 智能体将 "preferences.txt" 写入 /memories/ 路径
# res =agent1.invoke({
#     "messages": [{"role": "user", "content": "我最爱的水果是草莓, 请把我的偏好保存在/memories/preferences.txt"}]
# }, config=config1)
# res = agent.invoke(
#     {
#         'messages': [HumanMessage("调用工具写入一个文件，文件名为:测试.txt, 内容为: '测试'")]
#     }
# )

# res = agent.invoke(
#     {
#         'messages': [
#             HumanMessage("调用工具写入一个文件，文件名为:测试.txt, 内容为: '你好帅'"),
#             HumanMessage('调用工具读取名为测试.txt的文件，告诉我里面的内容')
#         ]
#     },
# )


config2 = {"configurable": {"thread_id": '2'}}

# res = agent1.invoke({
#     "messages": [{"role": "user", "content": "请从/memories/获取我最爱的水果是什么?"}]
# }, config=config2)

print("\n[线程 B] 新会话，尝试读取文件...")
result_b = agent1.invoke(
    {
        "messages": [{
            "role": "user",
            "content": """
                请尝试读取以下两个文件，并告诉我结果：
                1. /draft.txt
                2. /memories/user/preferences.txt
                """
        }]
    },
    config=config2
)

print(f"\n[线程 B] 响应：")
print(result_b["messages"][-1].content)

