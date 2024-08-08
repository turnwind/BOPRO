import os
import json
import re
import ast
import numpy as np
import matplotlib.pyplot as plt
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

with open("prompt.txt","r",encoding="utf-8") as file:
    says = file.read()

def Getvalue(ch,str):
    s = str.find(ch)
    e = str[s+1:].find(ch)
    return str[s+1:s+e+1]

def warm_strat(num):
    pt = (
        "You need to assume {} initial points for an optimization problem, and the objective function corresponding to the initial points should be as small as possible."
        "The objective function of this task is a binary quadratic function."
        "x1 must be within the range of [-5, 5]. x2 must be within the range of [-5, 5]."
        "According to the experience of previous tasks, the minimum value of the objective function may be around x1, x2 = 3, 3."
        "Please do not provide duplicate values."
        "Please give your answer and format your output as follows: *[],[],[],...,[]*").format(num)
    datas = Getvalue("*",chat(pt))
    datas = "[" + datas + "]"
    datas =  ast.literal_eval(datas)
    print(datas)

def candidate_sampling(history,num):
    # history = [{params: [1,2], loss: 3}, {params: [2,3], loss: 4}]
    pt = (
        "Based on the previous optimization results {}, you need to provide {} candidate points for the next optimization."
        "The objective function of this task is a binary quadratic function."
        "x1 must be within the range of [-5, 5]. x2 must be within the range of [-5, 5]."
        "Please do not provide duplicate values."
        "Please give your answer and format your output as follows: *[],[],[],...,[]*").format(json.dumps(history),num)
    datas = Getvalue("*",chat(pt))
    datas = "[" + datas + "]"
    datas =  ast.literal_eval(datas)
    print(datas)
history = [{"params": [1,2], "loss": 3}, {"params": [2,3], "loss": 4}]
candidate_sampling(history,5)


# while True:
#     str = chat(says)
#     s = str.find("{")
#     e = str.find("}")
#     str = str[s:e+1]
#     data = json.loads(str)
#     numbers = list(data.values())
#     loss = obj(numbers[0][0],numbers[0][1])
#     loss = np.round(loss,2)
#     text  = str + " - [\"loss\": {}]".format(loss)
#     says = (says[:-198] + "\n"+ text  +says[-198:])
#     print("user: "+says)
#     if loss == 0:
#         break
# warm_strat(5)