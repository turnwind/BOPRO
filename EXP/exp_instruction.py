import pandas as pd
import json
import re
import ast
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from tqdm import tqdm
from pydantic import BaseModel

# Load the data
datas = pd.read_csv("./Data/gsm_data/gsm_train.tsv", sep="\t")


class Step(BaseModel):
    explanation: str
    output: str

class ListResponse(BaseModel):
    steps: list[Step]
    final_answer: list[str]

class FloatResponse(BaseModel):
    steps: list[Step]
    final_answer: float

class FlistResponse(BaseModel):
    steps: list[Step]
    final_answer: list[float]
    
client = OpenAI(
    base_url="https://api.chatanywhere.tech/v1"
)

def chat(says,outformat):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are an experienced optimization master."},
            {"role": "user", "content": says},
        ],
        response_format=outformat,
    )
    message = completion.choices[0].message
    if message.parsed:
        #print(message.parsed.steps)
        print(message.parsed.final_answer)
        return message.parsed.final_answer
    else:
        print(message.refusal)
        chat(says)
    
def Getvalue(ch,str):
    s = str.find(ch)
    e = str[s+1:].find(ch)
    return str[s+1:s+e+1]

meta_prompt = "Instructions that can improve the accuracy and effectiveness of the model often guide the model to gradually reason, analyze in detail, think structurally, and collaborate to solve problems."
ini_prompt = [{"instruction": "Let’s think step by step."},{"instruction": "Let’s think carefully about the problem and solve it together."},{"instruction":" Break this down"}]

def Obj(INS):
    s = 0
    for i in range(50):
        quesion = datas.iloc[i,0] + INS
        answer = chat(quesion,FloatResponse)
        if answer == datas.iloc[i,1]:
            s += 1
    return s/50
    
    
def warm_strat(num):
    pt = (
        "You need to generate {} instructions to enhance the performance of LLM."
        "This is a summary of experiences that may help you."
        "{}"
        "Common instructions include: "
        "{}"
        "Please give your recommended instructions.").format(num,json.dumps(meta_prompt),meta_prompt)
    datas = chat(pt,ListResponse)
    return datas

def candidate_sampling(history,num):
    pt = (
        "Based on the previous optimization results {}, you need to provide {} candidate instruction for the next optimization."
        "Please do not provide duplicate values."
        "This is a summary of experiences that may help you."
        "{}"
        "Please give your recommended instructions.").format(json.dumps(history),num,meta_prompt)
    datas = chat(pt,ListResponse)
    return datas

def SurrogateModel(history,samples):
    data_pred = []
    pt = ("You need to estimate the accuracy of this instruction on the gsm(A dataset of mathematical question and answer.) dataset."
        "This is a summary of experiences that may help you."
        "{}"
        "Below is the historical evaluation data"
        "{}"
        "Please guess the accuracy for these instructions as follow:"
        "{}").format(json.dumps(history),meta_prompt,samples)
    datas = chat(pt,FlistResponse)
    return datas


if __name__ == "__main__":
    ### config
    numiters = 10        # number of iters for BO
    numinipoint = 2      # number of ini points
    numsamples = 2       # number of sampled points
    
    arrloss = [0]
    ins = warm_strat(2)
    datas = []
    for i in range(numinipoint):
        s = Obj(ins[i])
        arrloss[0] = max(arrloss[0],s)
        datas.append({"instruction": ins[i], "accuracy": s})

    for i in range(10):
        ins_sp = candidate_sampling(ins,2)
        s = SurrogateModel(datas,ins_sp)
        index = s.index(max(s))
        datas.append({"instruction": ins_sp[index], "accuracy": Obj(ins_sp[index])})
    
    
    print("The final result is: ",max(arrloss))
    
    for i in range(len(arrloss)):
        arrloss[i] = max(arrloss[:i+1])
    plt.plot(arrloss)
    plt.show()
    