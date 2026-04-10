import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse  # 用于流式输出
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# --- 1. 加载配置 ---
load_dotenv()  # 读取 .env 文件中的环境变量


# --- 2. 初始化 FastAPI ---
app = FastAPI(title="Qwen API Service (LangChain)")

# --- 3. 配置 Qwen 模型 (关键!) ---
# 使用 langchain_openai 的 ChatOpenAI 类，但指向阿里云的 endpoint
llm = ChatOpenAI(
    model="qwen-plus",  # 可选: qwen-turbo, qwen-max 等
    api_key=os.getenv("OPENAI_API_KEY"),  # 从环境变量读取
    base_url=os.getenv("OPENAI_API_BASE"),  # 阿里云兼容端点
    temperature=0.7,
    streaming=True  # 开启流式输出，提升用户体验
)

# --- 4. 定义请求和响应的数据结构 ---
class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful assistant."  # 默认系统提示词

class ChatResponse(BaseModel):
    reply: str

# --- 5. 创建 API 路由 ---
@app.post("/chat2", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """非流式聊天接口，等待完整回答后返回"""
    try:
        # 构建 LangChain 的消息格式
        messages = [
            SystemMessage(content=request.system_prompt),
            HumanMessage(content=request.message)
        ]
        # 调用模型
        response = llm.invoke(messages)
        return ChatResponse(reply=response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat2/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """流式聊天接口，实时返回生成的内容"""
    async def generate():
        messages = [
            SystemMessage(content=request.system_prompt),
            HumanMessage(content=request.message)
        ]
        # 使用 .stream() 方法进行流式调用
        async for chunk in llm.astream(messages):
            # LangChain 的流式块是 AIMessageChunk 对象
            if chunk.content:
                yield chunk.content

    return StreamingResponse(generate(), media_type="text/plain")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app,host="localhost",port=8000)
