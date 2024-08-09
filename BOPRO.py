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
    flag  = 1 #Determine whether the output meets the format.
    while flag:
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
        if len(datas) == num:
            flag = 0
    return datas

def candidate_sampling(history,num):
    flag  = 1 #Determine whether the output meets the format.
    while flag:
        pt = (
            "Based on the previous optimization results {}, you need to provide {} candidate points for the next optimization."
            "The objective function of this task is a binary quadratic function."
            "x1 must be within the range of [-5, 5]. x2 must be within the range of [-5, 5]."
            "Please do not provide duplicate values."
            "Please give your answer and format your output as follows: *[],[],[],...,[]*").format(json.dumps(history),num)
        datas = Getvalue("*",chat(pt))
        datas = "[" + datas + "]"
        datas =  ast.literal_eval(datas)
        if len(datas) == num:
            flag = 0
    return datas
    
def SurrogateModel(history,samples):
    data_pred = []
    for i in range(len(samples)):
        flag  = 1 #Determine whether the output meets the format.
        while flag:
            pt = ("You are helping tune hyperparameters to minimize loss(>0)."
                "The objective function of this task is a binary quadratic function."
                "You need to guess the target function value for a given x based on historical evaluation data."
                "Below is the historical evaluation data, formatted as [Hyperparameters] - [loss]:"
                "{}"
                "Please guess the loss for params:{} and format your output as follows: *xx*").format(history,samples[i])
            loss = re.findall(r'-?\d+\.\d+|-?\d+', Getvalue("*",chat(pt)))
            if len(loss) == 1:
                flag = 0
                new_entry = {"params": samples[i], "loss": float(loss[0])}
                data_pred.append(new_entry)
    return data_pred



if __name__ == "__main__":
    numiters = 10
    inidatas = warm_strat(3)
    datas = []
    arrloss = []
    for i in inidatas:
        loss = obj(i[0],i[1])
        new_entry = {"params": [i[0], i[1]], "loss": loss}
        datas.append(new_entry)
        
    for i in range(numiters):
        samplers = candidate_sampling(datas,5)
        data_pred = SurrogateModel(datas,samplers)
        next_point = min(data_pred, key=lambda x: x['loss'])
        loss = obj(next_point["params"][0],next_point["params"][1])
        arrloss.append(loss)
        new_entry = {"params": next_point["params"], "loss": loss} 
        datas.append(new_entry)
    
    print("The final result is: ",min(arrloss))
    plt.plot(arrloss)
    plt.show()
      