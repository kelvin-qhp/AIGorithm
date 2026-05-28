# pip install langchain langchain-openai langgraph

import os
import json
from typing import List
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()
# ========== 1. 配置模型（使用阿里云百炼通义千问） ==========
model = ChatOpenAI(
    model="qwen-plus",
    openai_api_key=os.getenv("OPENAI_API_KEY"),  # 替换为你的 API Key
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    temperature=0.1,
)

# ========== 2. 定义结构化输出格式 ==========
class ProductExtractionResult(BaseModel):
    """商品信息抽取结果"""
    core_words: List[str] = Field(description="中心词列表，如产品名称、产品类型等主体词")
    attribute_words: List[str] = Field(description="属性词列表，如品牌、型号、规格、颜色、材质、功能等")
    category: str = Field(description="商品类别")
    confidence: float = Field(description="提取置信度", ge=0, le=1)

# ========== 3. 定义两个工具 ==========

@tool
def extract_core_words(product_text: str) -> str:
    """
    从商品信息中提取中心词（核心产品词）
    中心词包括：产品名称、产品类型、产品系列等主体词汇

    Args:
        product_text: 商品标题和描述的完整文本
    """
    prompt = f"""
            请从以下商品信息中提取中心词（核心产品词）：
            
            商品信息：{product_text}
            
            要求：
            1. 中心词是产品的核心主体，如"手机"、"笔记本电脑"、"连衣裙"等
            2. 排除品牌、规格、颜色等修饰词
            3. 如果有多个产品主体，全部提取
            
            请按以下JSON格式返回：
            {{"core_words": ["词1", "词2", ...]}}
            """
    # 这里用简单规则模拟，实际项目中会调用 LLM
    # 简化示例：根据关键词提取
    result = {"core_words": []}

    # 模拟提取逻辑（实际应由 LLM 完成）
    if "手机" in product_text:
        result["core_words"].append("手机")
    if "电脑" in product_text or "笔记本" in product_text:
        result["core_words"].append("笔记本电脑")
    if "大衣" in product_text or "外套" in product_text:
        result["core_words"].append("大衣")

    return json.dumps(result, ensure_ascii=False)


@tool
def extract_attribute_words(product_text: str) -> str:
    """
    从商品信息中提取属性词（特征描述词）
    属性词包括：品牌、型号、规格、颜色、材质、功能、尺寸等

    Args:
        product_text: 商品标题和描述的完整文本
    """
    prompt = f"""
        请从以下商品信息中提取属性词：
        
        商品信息：{product_text}
        
        要求：
        1. 提取所有描述产品特征的词汇
        2. 包括但不限于：品牌、型号、规格、颜色、材质、功能、尺寸、重量、容量
        3. 排除中心词本身
        
        请按以下JSON格式返回：
        {{"attribute_words": ["词1", "词2", ...]}}
        """
    # 简化示例：根据关键词提取
    result = {"attribute_words": []}

    # 模拟提取逻辑
    brand_keywords = ["小米", "华为", "苹果", "三星", "联想"]
    spec_keywords = ["16GB", "512GB", "5G", "OLED", "骁龙"]
    material_keywords = ["羊毛", "棉", "皮革", "丝绸", "聚酯纤维"]
    color_keywords = ["黑色", "白色", "驼色", "灰色", "蓝色"]

    for word in brand_keywords + spec_keywords + material_keywords + color_keywords:
        if word in product_text and word not in result["attribute_words"]:
            result["attribute_words"].append(word)

    return json.dumps(result, ensure_ascii=False)


# ========== 4. 创建 Agent ==========
agent = create_agent(
    model=model,
    tools=[extract_core_words, extract_attribute_words],
    system_prompt="""你是一个专业的电商商品信息分析助手。你的任务是从商品标题和描述中提取信息：
            工作流程：
            1. 首先使用 extract_core_words 工具提取商品的核心主体词
            2. 然后使用 extract_attribute_words 工具提取商品的属性特征词
            3. 根据提取结果，判断商品类别
            4. 最终给出结构化的分析结果
            
            注意：工具返回的是 JSON 字符串，你需要解析后使用。
            """,
    response_format=ProductExtractionResult,
)


# ========== 5. 执行抽取 ==========
def extract_product_info(product_title: str, product_description: str = "") -> ProductExtractionResult:
    """
    抽取商品的中心词和属性词
    """
    # 合并商品信息
    product_text = f"标题：{product_title}\n"
    if product_description:
        product_text += f"描述：{product_description}"

    # 调用 Agent
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"请分析以下商品信息，提取中心词和属性词：\n{product_text}"
            }
        ]
    })

    # 返回结构化结果
    return result["structured_response"]


# ========== 6. 测试示例 ==========
if __name__ == "__main__":
    # 测试商品 1：智能手机
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

    print("=" * 70)
    print("商品 1：智能手机")
    print("=" * 70)
    result1 = extract_product_info(product_1["title"], product_1["description"])
    print(f"中心词：{', '.join(result1.core_words)}")
    print(f"属性词：{', '.join(result1.attribute_words)}")
    print(f"类别：{result1.category}")
    print(f"置信度：{result1.confidence}")

    print("\n" + "=" * 70)
    print("商品 2：羊毛大衣")
    print("=" * 70)
    result2 = extract_product_info(product_2["title"], product_2["description"])
    print(f"中心词：{', '.join(result2.core_words)}")
    print(f"属性词：{', '.join(result2.attribute_words)}")
    print(f"类别：{result2.category}")
    print(f"置信度：{result2.confidence}")