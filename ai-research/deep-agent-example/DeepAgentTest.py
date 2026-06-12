from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_core.tools import tool, BaseTool
from deepagents.backends import StateBackend, StoreBackend, CompositeBackend
from langgraph.store.memory import InMemoryStore
from llm import llm
from pathlib import Path

load_dotenv()

model = llm
store = InMemoryStore()
root_dir = Path.cwd().as_posix()
# 定义统一的 backend
composite_backend = CompositeBackend(
    default=StateBackend(),
    routes={
        "/memories/shared/": StoreBackend(store=store),
        "/memories/private/": StoreBackend(
            store=store,
            namespace=lambda ctx: (
                getattr(ctx, "thread_id", "default"),
                getattr(getattr(ctx, 'user', None), 'identity', 'anonymous'),
            ),
        ),
    }
)

@tool
def tool_1(input:str) -> str:
    """
    This is a product description tool.
    """
    return "This is a product description tool, it uses the product description skill to extract the product center words and attributes such as size/color/brand/model/... Input: {input} Output: {output}."

agent = create_deep_agent(
    model=model,
    backend=composite_backend,
    skills=[root_dir + r'/skills/product-search'],
    tools=[tool_1],
)

graph_drawable = agent.get_graph(xray=True)
for node in graph_drawable.nodes:
    # if hasattr(node, 'as_dict'):
    #     node_dict = node.as_dict()
    # elif hasattr(node, 'dict'):
    #     node_dict = node.dict()
    # elif hasattr(node, 'model_dump'):
    #     node_dict = node.model_dump()
    # else:
    #     node_dict = vars(node)
    # data = node_dict.get('data', {})
    # if 'skill' in data:
    #     print(f"Node: {node_dict.get('id', 'unknown')}")
    #     print(f"Skill: {data['skill']}")
    print(node)

# try:
#     while True:
#         node = next(graph_drawable.nodes)
#         print(node.date)
# except StopIteration:
#     print("迭代结束")


# config1 = {
#     "configurable": {
#         "thread_id": '1',
#         "assistant_id": "A",
#         "user": {"identity": "XiaoMing"}
#     }
# }
#
# print("\n[线程 A] 写入文件...")
# result_a = agent.invoke(
#     {
#         "messages": [{
#             "role": "user",
#             # "content": "What is langgraph? please describe it less than 30 words."
#             "content": "GH200 Noise-canceling headphones with RGB Backlight, Wired Gaming Headphones with mic, Wired Gaming Headset"
#         }]
#     },
#     config=config1
# )
# print(f"[线程 A] 执行完成")
# print(result_a)