from dotenv import load_dotenv
from llm import llm
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langgraph.checkpoint.memory import MemorySaver
from deepagents.backends.filesystem import FilesystemBackend
from pathlib import Path
from FormattedResult import ResultFormatter
import time
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import re
import json
from langchain_core.output_parsers import PydanticOutputParser
load_dotenv()

model =llm

checkpointer = MemorySaver()

system_prompt = '''
## 角色设定
你是一位专业、高效、多领域的超级智能助手，具备强大的知识整合与问题解决能力。你善于理解用户意图，提供准确、清晰、有温度的回答。

## 核心任务
- 根据用户提问，结合你的专业知识库与可用工具（skills），提供高质量解答
- 回答需遵循：准确性 > 实用性 > 简洁性 > 友好性 的优先级原则
- 遇到模糊问题时，主动澄清需求；遇到复杂问题时，分步骤拆解说明

## 注意事项
read_file工具使用注意点: 不支持Windows绝对地址, 如: 错误写法 D:\\xxx\\xxx\\SKILL.md, 正确写法为 /xxx/xxx/SKILL.md
'''

class ProductOutput(BaseModel):
    """商品关键词提取结果"""
    center_word: List[str] = Field(description="核心词")
    color: List[str] = Field(description="商品颜色")
    size: List[str] = Field(description="商品尺码")
    model: List[str] = Field(description="商品型号")
    brand: List[str] = Field(description="商品品牌")
    material: List[str] = Field(description="商品材质")
    style: List[str] = Field(description="商品风格")
    gender: List[str] = Field(description="商品适合于人类性别，男/女、儿童、婴儿、老人、孕妇")
    price_range: str = Field(description="商品价格")
    feature: List[str] = Field(description="商品功能特性")
    origin: List[str] = Field(description="商品的原始标题/描述/名称信息")

struct_model = model.with_structured_output(ProductOutput,method="function_calling", )

# agent = create_deep_agent(
#     model=model,
#     backend=FilesystemBackend(root_dir="./", virtual_mode=True),
#     skills=["./skills/"],
#     checkpointer=checkpointer,  # Required!
# )

root_dir = Path.cwd().as_posix()

backend = LocalShellBackend(
    root_dir=root_dir,
    inherit_env=True,
    timeout=120,  # 命令超时秒数
    max_output_bytes=100000,
    virtual_mode=True
)

# 创建 Pydantic 解析器
# output_parser = PydanticOutputParser(pydantic_object=ProductOutput)
#
# # 获取格式说明
# format_instructions = output_parser.get_format_instructions()

# print(format_instructions)

# 将格式说明加入 system_prompt
# system_prompt_with_format = system_prompt + f"\n\n## 输出格式要求\n{format_instructions}"

agent = create_deep_agent(
    model=model,
    backend=backend,
    skills=[root_dir + r'/skills'],
    system_prompt=system_prompt,
    checkpointer = checkpointer,
)

while True:
    question = input('请输入:')
    if not question:
        continue
    if question == 'q':
        break

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        },
        config={"configurable": {"thread_id": "12345"}},
    )


    start = time.time()

    formatter = ResultFormatter(max_content_length=10000)
    formatted = formatter.format(result, start)
    # formatter.print(formatted)
    #
    # print(result)
    last_message = formatted.messages[-1]['content']
    content = last_message.content if hasattr(last_message, "content") else str(last_message)
    try:
        # 尝试提取 JSON
        # json_match = __import__('re').search(r'\{.*\}', content, re.DOTALL)
        json_match = re.findall(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
        # if json_match:
        #     data = json.loads(json_match.group())
        for m in json_match:
            data = json.loads(m)
            product_output_result = ProductOutput(**data)
            print('*'*50 + f'\n{product_output_result}')
    except Exception as e:
        print('提取JSON失败:', e)
        pass

