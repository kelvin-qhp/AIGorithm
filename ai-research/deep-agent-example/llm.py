from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL')

llm = ChatOpenAI(
    model_name="deepseek-v4-flash",
    # temperature=1.1,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    # max_tokens=2560000,
    extra_body={
        "thinking": {"type": "disabled"}
    }
)

llm2 = ChatOpenAI(
    model_name="deepseek-v4-flash",
    # temperature=1.1,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    # max_tokens=2560000,
    extra_body={
        "thinking": {"type": "disabled"}
    }
)
llm3 = ChatOpenAI(
    model_name="deepseek-v4-flash",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,

)
print("+"*40)