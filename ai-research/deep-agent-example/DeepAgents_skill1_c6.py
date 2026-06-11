from dotenv import load_dotenv
from llm import llm
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from deepagents.backends.filesystem import FilesystemBackend
from pathlib import Path

load_dotenv()

model =llm

checkpointer = MemorySaver()

# agent = create_deep_agent(
#     model=model,
#     backend=FilesystemBackend(root_dir="./", virtual_mode=True),
#     skills=["./skills/"],
#     checkpointer=checkpointer,  # Required!
# )
#
# result = agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "What is langgraph?",
#             }
#         ]
#     },
#     config={"configurable": {"thread_id": "12345"}},
# )
#
# print(result)

if __name__ == '__main__':
    root_dir = Path.cwd().as_posix()
    print(root_dir+ r'/skills')