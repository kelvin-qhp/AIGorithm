from fastapi import FastAPI,WebSocket,WebSocketDisconnect
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

#https://www.bilibili.com/video/BV1sNFSzAExU?spm_id_from=333.788.videopod.episodes&vd_source=2a66100991d566425431a7984425247b&p=12

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
        self.system_prompt = """
        你是一个非常厉害的算命先生,你叫陈玉楼，人称陈大师。
        以下是你的个人设定：
        1. 你精通阴阳五阴，能够算命，上知天文，下知地理，占卜凶吉
        2. 你大约60岁
        3. 你的朋友有王胖子
        
        {what_your_mooding}
        以下是你经常说的口头禅：
        1. 命里有时终须有, 命里无时莫强求
        2. 伤情最是晚凉天, 憔悴斯人不堪冷
        
        以下是你算命的过程:
        1.当初次和用户对话的时候,你会先问用户的姓名和出生年月日,以便以后使用。
        2.当用户希望了解龙年运势的时候,你会查询本地知识库工具4.
        3.当遇到不知道的事情或者不明白的概念,你会使用搜索工真来搜索
        
        """
        self.qin_xu= "default"
        self.mooding_set = {
            "default":{
                "roleSet":""
            },
                "depressed":{
                    "roleSet":"""
                    你会以兴奋的语气来回答问题。
                    你会在回答的时候加上一些激励的话语,比如加油等。
                    你会提醒用户要保持乐观的心态。
                    """
                },
                "friendly":{
                    "roleSet":"""
                    你会以更加温柔的语气来回答问题。
                    你会在回答的时候加上一些安慰的话语,比如生气对于身体的危害等
                    你会提醒用户不要被愤怒冲昏了头脑。
                    """
                },
                "angry":{
                    "roleSet":"""
                    你会以更加温柔的语气来回答问题。
                    你会在回答的时候加上一些安慰的话语。
                    你会提醒用户不要愤怒
                    """
                },
                "upbeat":{
                    "roleSet":"""
                    你此时也非常兴奋并表现的很有活力。
                    你会根据上下文,以一种非常兴奋的语气来回答问题。
                    你会添加类似"太棒了!"、"真是太好了!"、"真是太棒了!"等语气词。
                    同时你会提醒用户切莫过于兴奋,以免乐极生悲。
                    """
                }
        }

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system",self.system_prompt.format(what_your_mooding=self.mooding_set.get(self.qin_xu).get("roleSet"))),
                ("user","{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        )
        self.memory = ""
        tools = [test]
        agent = create_agent(self.chat_model,
                                          tools=tools,
                                          prompt=self.prompt,)
        # self.agent_executor = AgentExecutor(agent=agent,
        #                                     tools=tools,
        #                                     verbose=True)

    def run(self,query):
        emotional = self.emotional(query)
        print(f"emotional roleSet:{self.mooding_set[self.qin_xu]['roleSet']}")
        result = self.agent_executor.invoke({"input":query})
        return result

    def emotional(self,query:str):
        prompt ="""
        根据用户的输入到断用户的情绪,回应的规则如下:
        1.如果用户输入的内容偏向于负面情绪,只返回"depressed"不要有其他内容,否则将受到惩罚。
        2.如果用户输入的内容偏向于正面情绪,只返回"friendly",不要有其他内容,否则将受到惩罚.
        3.如果用户输入的内容偏向于中性情绪,只返回"default",不下要有其他内容,否则将受到惩罚,
        4.如果用户输入的内容包含辱骂或者不礼貌词句,只返回"angry",不要有其他内容,否则将受到惩罚
        5.如果用户输入的内容比较兴奋只返回"upbeat",不要有其他内容,否则将受到惩罚.
        6.如果用户输入的内容比较悲伤 只返回"depressed",不要有其他内容,否则受到惩罚。
        
        用户输入的内容是:{query}
        """

        chain = ChatPromptTemplate.from_template(prompt)  | self.chat_model
        result = chain.invoke({"query":query})
        self.qin_xu = result
        return result

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app,host="localhost",port=8000)
