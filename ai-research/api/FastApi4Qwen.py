from fastapi import FastAPI,WebSocket,WebSocketDisconnect
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent,AgentExecutor,tool
from dotenv import load_dotenv
import os


# --- 1. 加载配置 ---
load_dotenv()  # 读取 .env 文件中的环境变量

app = FastAPI()
@tool
def test():
    """test tools"""
    return "test"
@app.get("/")
def getTest():
    return {"hellow":"world"}

@app.post("/chat")
def chat(query:str):
    master = Master()
    return master.run(query)

@app.websocket("/ws")
async def websocket_endpoint(websocket:WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Send message:{data}")
    except WebSocketDisconnect:
        print("Connection is close.")
        await websocket.close()



class Master:
    def __init__(self):
        self.chat_model = ChatOpenAI(model_name="qwen-plus",
                                     openai_api_key=os.getenv("OPENAI_API_KEY"),
                                     openai_api_base=os.getenv("OPENAI_API_BASE"),
                                     temperature=0,
                                     streaming=True)
        self.memory_key = "chat_history"
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system","you are a assisant"),
                ("user","{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        )
        self.memory = ""
        tools = [test]
        agent = create_openai_tools_agent(self.chat_model,
                                          tools=tools,
                                          prompt=self.prompt,)
        self.agent_executor = AgentExecutor(agent=agent,
                                            tools=tools,
                                            verbose=True)

    def run(self,query):
        result = self.agent_executor.invoke({"input":query})
        return result





if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app,host="localhost",port=8000)
