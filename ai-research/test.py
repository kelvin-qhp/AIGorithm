import numpy as np

import requests
import pickle

import torch
from openai import OpenAI
import json
import re
from util.utils import load_dict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification,pipeline


# print(np.log(np.ones(7)/7))
# arr = [1,2,3,4,5,6]
# with open("./data/input/hmm_dict/pi.txt", mode="wb+") as f:
#     pickle.dump(arr,f,)

# with open("./data/input/hmm_dict/pi.txt", mode="rb") as f:
#     load_data = pickle.load(f)
#     print("*"*10)
#     print(load_data)
#

# m1 = np.array([[1, 2], [3, 4],[5,6]])
# m2 = [[10],[10],[10]]
# m3 = [[100,1000]]
# r1 = m1 + m2
# print(r1+ m3)

#
# a1 = np.arange(12).reshape(3,4)
# print(a1)
# print(a1.T)
# print(a1.argmax(axis=0))
# print(a1.argmax(axis=1))
# print(np.ravel(a1,order='F'))

# print(np.random.multinomial(20, [1 / 6.] * 6, size=2))
# print(np.arange(9.).reshape(3, 3))

def executeDeepSeek():
    client = OpenAI(
        api_key="ollama",
        base_url="https://ollama.globalsources.com/v1/",
    )

    system_prompt = """
    The user will provide some exam text. Please parse the "question" and "answer" and output them in JSON format. 
    
    EXAMPLE INPUT: 
    Which is the highest mountain in the world? Mount Everest.
    
    EXAMPLE JSON OUTPUT:
    {
        "question": "Which is the highest mountain in the world?",
        "answer": "Mount Everest"
    }
    """

    user_prompt = "Which is the longest river in the world? The Nile River."

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

    response = client.chat.completions.create(
        model="deepseek-r1:14b",
        messages=messages,
        response_format={
            'type': 'json_object'
        }
    )

    print(json.loads(response.choices[0].message.content))


# executeDeepSeek()

# url = "http://ollama.globalsources.com/api/generate"
# url = "http://ollama.globalsources.com/v1/completions"
# payload = {
#     "prompt": "Hello, how are you?",
#     "max_tokens": 500
# }
# response = requests.post(url, json=payload)
# print(response.json())

if __name__ == '__main__':
    # file_path = 'D:\src\\ai-research\data\input\gemini\output_P1_prediction-model-2025-06-18T09_38_09.601103Z_predictions.jsonl'
    # file_path = 'D:\src\\ai-research\data\input\gemini\output_P1_prediction-model-2025-06-18T10_18_16.006977Z_predictions.jsonl'
    # # dict = load_dict(file_path)
    # # print(dict)
    # i = 0
    # json_pattern = re.compile(r'\'\'\'json.\'\'\'', re.DOTALL)
    # with open(file_path, "r", encoding="utf-8") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         lineJson = json.loads(line)
    #         i +=1
    #         outputStr = lineJson['response']['candidates'][0]['content']['parts'][0]['text']
    #         outputStr = outputStr.strip("`").strip('json\n')
    #         print(i,outputStr)

    a = torch.randn(1,3)
    print(a,a.shape,a.dim())
    b = torch.squeeze(a,0)
    print(b,b.shape,b.dim())


    c= torch.unsqueeze(a,1)
    print(c,c.shape,c.dim())
    print(torch.squeeze(c,0),c.dim())
    print(torch.squeeze(c,1),c.dim())

    x = torch.randn(10,3)
    y= x.sum(1)
    print(x,y.shape,y)

    print("*"*100)
    m = torch.randn(3,5,8)
    mh = m[1:][:,0,:]
    print(m.shape,m)
    print(mh.shape,mh)
