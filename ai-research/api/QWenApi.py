import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
api_url = os.getenv("OPENAI_API_BASE") + "chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": "qwen-plus", # 可选的模型：qwen-turbo, qwen-plus, qwen-max等
    "messages": [
        {"role": "system", "content": "你是一个乐于助人的助手。"},
        {"role": "user", "content": "你好，请介绍一下自己。"}
    ]
}

# 发送请求
response = requests.post(api_url, headers=headers, json=data)
print(response.json()["choices"][0]["message"]["content"])