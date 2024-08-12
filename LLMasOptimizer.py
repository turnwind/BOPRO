import os
import json
import re
import ast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(
    base_url="https://api.chatanywhere.tech/v1"
)

def chat(says):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": says,
            }
        ],
        model="gpt-4o-2024-05-13",
    )
    print("bots: "+chat_completion.choices[0].message.content)
    return  chat_completion.choices[0].message.content

def obj(x,y):
    return (x-2)**2+(y-3)**2

def Getvalue(ch,str):
    s = str.find(ch)
    e = str[s+1:].find(ch)
    return str[s+1:s+e+1]

def Optimizer(history):
    flag  = 1 #Determine whether the output meets the format.
    while flag:
        pt = ("You are helping tune hyperparameters to minimize loss(>0)."
            "The objective function of this task is a binary quadratic function."
            "x1 must be within the range of [-5, 5]. x2 must be within the range of [-5, 5]."
            "I want you to recommend a new configuration that can minimize the loss."
            "Hera are your historical trial data which can help you select next point, formatted as [Hyperparameter] - [loss]:"
            "{}"
            "Please do not provide parameters that have already been tried!"
            "As short as possible, do not provide too much information."
            "Please give you answer and format your output as follows : *[]*").format(json.dumps(history))
        datas = Getvalue("*",chat(pt))
        datas = "[" + datas + "]"
        try:
            datas =  ast.literal_eval(datas)
        except Exception as e:
            continue
        if len(datas) == 1:
            flag = 0
    return datas

if __name__ == "__main__":
    datas = []
    
    for i in range(13):
        point = Optimizer(datas)[0]
        datas.append({"params": point, "loss": obj(point[0],point[1])})
    
    x_params = [item['params'][0] for item in datas]
    y_params = [item['params'][1] for item in datas]
    # 保存loss结果到LLM.json文件,追加写入
    data = []
    for item in datas:
        data.append(item["loss"])

    # 打开json文件
    with open("LLM.json", "r") as f:
        data1 = json.load(f)
        data1["loss"].append(data)
    # 写入json文件
    with open("LLM.json", "w") as f:
        json.dump(data1, f)
    
