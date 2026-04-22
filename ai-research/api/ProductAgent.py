import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents import  create_agent
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

# ========== 1. 定义数据结构 ==========
class ProductKeywords(BaseModel):
    """商品关键词提取结果"""
    core_words: List[str] = Field(description="核心词列表，如产品名称、产品类型等主体词")
    attribute_words: List[str] = Field(description="属性词列表，如品牌、型号、规格、颜色、材质、功能等特征词")
    category: str = Field(description="商品所属类别，如：电子产品、服装、食品等")
    confidence: float = Field(description="置信度，0-1之间", ge=0, le=1)

# ========== 2. 定义工具 ==========
@tool
def analyze_product_title(title: str) -> str:
    """
    分析商品标题，提取关键信息
    Args:
        title: 商品标题文本
    """
    prompt = f"""
    分析以下商品标题，提取：
    1. 核心词：产品的主体名称
    2. 属性词：描述产品的特征、规格、品牌等
    
    商品标题：{title}
    
    请按格式返回：
    核心词：xxx
    属性词：xxx, xxx, xxx
    """
    return prompt

@tool
def extract_from_description(description: str) -> str:
    """
    从商品描述中提取详细的产品属性
    Args:
        description: 商品描述文本
    """
    prompt = f"""
    从以下商品描述中提取关键的产品属性和特征：
    
    商品描述：{description}
    
    请列出：
    1. 核心产品名称
    2. 品牌信息
    3. 规格参数（尺寸、重量、容量等）
    4. 材质/成分
    5. 功能特点
    6. 适用场景
    """
    return prompt

@tool
def detect_category(product_text: str) -> str:
    """
    判断商品所属类别
    Args:
        product_text: 商品完整文本（标题+描述）
    """
    categories = [
        "电子产品", "服装鞋包", "家居用品", "食品饮料",
        "美妆个护", "母婴用品", "运动户外", "图书音像",
        "玩具乐器", "汽车用品", "宠物用品", "医疗器械"
    ]

    prompt = f"""
    根据以下商品信息，从给定的类别中选择最匹配的一个：
    
    商品信息：{product_text}
    
    可选类别：{', '.join(categories)}
    
    只返回类别名称，不要有其他内容。
    """
    return prompt

# ========== 3. 创建主流程 ==========
class ProductKeywordExtractor:
    """商品关键词提取器"""

    def __init__(self, api_key: str = None, api_base: str = None, model_name: str = "qwen-plus"):
        """
        初始化提取器
        Args:
            api_key: 阿里云百炼 API Key
            api_base: API 基础地址
            model_name: 使用的模型名称
        """
        # 配置 LLM
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            openai_api_base=api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.1,  # 较低温度保证输出稳定性
        )

        # 创建解析器
        self.parser = PydanticOutputParser(pydantic_object=ProductKeywords)

    def extract_by_prompt_chain(self, product_title: str, product_description: str = "") -> ProductKeywords:
        """
        方式一：使用 Prompt Chain 直接提取（推荐，更稳定）
        """
        # 合并商品信息
        product_text = f"标题：{product_title}\n"
        if product_description:
            product_text += f"描述：{product_description}"

        # 构建提示词
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的电商商品信息分析专家。你的任务是从给定的商品信息中提取核心词和属性词。

            核心词：指产品的核心主体名称，如"手机"、"连衣裙"、"笔记本电脑"等。
            属性词：指描述产品特征的词汇，包括品牌、型号、规格、颜色、材质、功能、尺寸、重量等。

            请严格按照JSON格式返回结果。"""),
            ("user", "商品信息：\n{product_text}\n\n{format_instructions}")
        ])

        # 创建链
        chain = prompt | self.llm | self.parser

        # 执行
        result = chain.invoke({
            "product_text": product_text,
            "format_instructions": self.parser.get_format_instructions()
        })

        return result

    def extract_by_agent(self, product_title: str, product_description: str = "") -> Dict[str, Any]:
        """
        方式二：使用 Agent 方式（多步骤分析，更灵活）
        """
        # 准备工具列表
        tools = [analyze_product_title, extract_from_description, detect_category]

        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个电商商品信息分析助手。你需要：
                1. 使用工具分析商品标题和描述
                2. 从分析结果中提取核心词和属性词
                3. 判断商品类别
                
                最终返回结构化的分析结果。"""),
            ("human", "{input}"),
            # ("placeholder", "{agent_scratchpad}")
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        system_prompt = """你是一个电商商品信息分析助手。你需要：
                    1. 使用工具分析商品标题和描述
                    2. 从分析结果中提取核心词和属性词
                    3. 判断商品类别
                    
                    最终返回结构化的分析结果。
                """
        # 创建 Agent
        agent = create_agent(model=self.llm, tools=tools,system_prompt=system_prompt)
        # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # 执行 Agent
        product_text = f"标题：{product_title}\n"
        if product_description:
            product_text += f"描述：{product_description}"

        # result = agent.invoke({
        #     "input": f"请分析以下商品信息，提取核心词、属性词和类别：\n{product_text}"
        # })
        result = agent.invoke( HumanMessage(f"请分析以下商品信息，提取核心词、属性词和类别：\n{product_text}"))
        return result

    def batch_extract(self, products: List[Dict[str, str]]) -> List[ProductKeywords]:
        """
        批量提取多个商品的关键词
        Args:
            products: 商品列表，每个元素包含 title 和 description
        """
        results = []
        for product in products:
            result = self.extract_by_prompt_chain(
                product_title=product.get("title", ""),
                product_description=product.get("description", "")
            )
            results.append(result)
        return results

# ========== 4. 使用示例 ==========
def main():
    # 创建提取器
    extractor = ProductKeywordExtractor(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_API_BASE"),
        model_name="qwen-plus"  # 使用通义千问 plus 版本
    )

    # 测试商品 1：手机
    product_1 = {
        "title": "小米14 Pro 5G智能拍照手机 徕卡光学镜头 16GB+512GB 黑色",
        "description": """小米14 Pro搭载第三代骁龙8移动平台，配备徕卡光学Summilux镜头，
        支持可变光圈，2K超视感屏，4880mAh大电池，支持120W有线快充和50W无线快充。
        适合摄影爱好者和商务人士使用。"""
    }

    # 测试商品 2：女装
    product_2 = {
        "title": "韩版宽松羊毛大衣 中长款加厚保暖 气质驼色 S码",
        "description": """这款羊毛大衣采用90%澳洲绵羊毛，10%山羊绒混纺面料，
        具有极佳的保暖性和舒适度。H型版型设计，不挑身材，适合各种体型。
        经典驼色，百搭显气质。可机洗，不易起球。"""
    }

    print("=" * 60)
    print("商品 1：智能手机")
    print("=" * 60)
    result1 = extractor.extract_by_prompt_chain(
        product_title=product_1["title"],
        product_description=product_1["description"]
    )
    print(f"核心词：{', '.join(result1.core_words)}")
    print(f"属性词：{', '.join(result1.attribute_words)}")
    print(f"类别：{result1.category}")
    print(f"置信度：{result1.confidence}")

    print("\n" + "=" * 60)
    print("商品 2：羊毛大衣")
    print("=" * 60)
    result2 = extractor.extract_by_prompt_chain(
        product_title=product_2["title"],
        product_description=product_2["description"]
    )
    print(f"核心词：{', '.join(result2.core_words)}")
    print(f"属性词：{', '.join(result2.attribute_words)}")
    print(f"类别：{result2.category}")
    print(f"置信度：{result2.confidence}")

    # 批量提取示例
    print("\n" + "=" * 60)
    print("批量提取示例")
    print("=" * 60)
    products = [product_1, product_2]
    batch_results = extractor.batch_extract(products)
    for i, res in enumerate(batch_results):
        print(f"\n商品 {i+1}:")
        print(f"  核心词: {res.core_words}")
        print(f"  属性词: {res.attribute_words[:5]}...")  # 只显示前5个

# ========== 5. 增强版：带缓存和重试机制 ==========
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential

class EnhancedProductKeywordExtractor(ProductKeywordExtractor):
    """增强版提取器，支持缓存和自动重试"""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def extract_with_retry(self, product_title: str, product_description: str = "") -> ProductKeywords:
        """带重试机制的提取"""
        return self.extract_by_prompt_chain(product_title, product_description)

    @lru_cache(maxsize=100)
    def extract_cached(self, product_title: str, product_description: str = "") -> ProductKeywords:
        """带缓存的提取（相同商品不重复调用 API）"""
        return self.extract_by_prompt_chain(product_title, product_description)

if __name__ == "__main__":
    # --- 1. 加载配置 ---
    load_dotenv()  # 读取 .env 文件中的环境变量

    # main()

    extractor = ProductKeywordExtractor(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_API_BASE"),
        model_name="qwen-plus"  # 使用通义千问 plus 版本
    )

    # 测试商品 1：手机
    product_1 = {
        "title": "小米14 Pro 5G智能拍照手机 徕卡光学镜头 16GB+512GB 黑色",
        "description": """小米14 Pro搭载第三代骁龙8移动平台，配备徕卡光学Summilux镜头，
        支持可变光圈，2K超视感屏，4880mAh大电池，支持120W有线快充和50W无线快充。
        适合摄影爱好者和商务人士使用。"""
    }

    result0 = extractor.extract_by_agent(product_title=product_1.get("title", ""),
                               product_description=product_1.get("description", ""))

    print(f"agent output：{result0}")

