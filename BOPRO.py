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
        model="gpt-4o-2024-08-06",
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
            "x1 must be within the range of [-5, 5]. x2 must be within the range of [-5, 5]"
            "According to the experience of previous tasks."
            "Please do not provide duplicate values."
            "Please give your answer and format your output as follows: *[],[],[],...,[]*").format(num)
        datas = Getvalue("*",chat(pt))
        datas = "[" + datas + "]"
        try:
            datas =  ast.literal_eval(datas)
        except Exception as e:
            continue
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
        try:
            datas =  ast.literal_eval(datas)
        except Exception as e:
            continue
        if len(datas) == num:
            flag = 0
    return datas

agent_prompts = {
    "optimistic": "You are an optimistic agent. Inclined to explore uncertainties.",
    "pessimistic": "You are a pessimistic agent. Inclined to avoid risks.",
    "Conservative": "You are a conservative agent. You weigh the pros and cons to make decisions.",
}
def AcquisitionFunction(history,pred):
    
    m = 2
    inipt = ("{}"
            "You need to reasonably select the most potential parameters from the predicted data table based on your own experience and historical data tables for the next optimization step."
            "We hope you can balance the relationship between exploration and exploitation to achieve the best results."
            "The following is a historical data table, with format [Hyperparameters] - [loss]:"
            "{}"
            "The following is a predicted data table, with format [Hyperparameters] - [predloss]:"
            "{}"
            "A brief summary of your reasons, as concise as possible."
            "Please provide 2 sets of parameter combinations that you consider optimal, in the format *[],[]*")
    res = {"optimistic":"","Conservative":"","pessimistic":""}
    datas = []
    for i in agent_prompts.keys():
        flag  = 1 #Determine whether the output meets the format.
        while flag:
            pt = inipt.format(agent_prompts[i],json.dumps(history),json.dumps(pred))
            res[i] = chat(pt)
            str = Getvalue("*",res[i])
            str = "[" + str + "]"
            try:
                data =  ast.literal_eval(str)
            except Exception as e:
                continue
            if len(data) == 2:
                datas.append(data)
                flag = 0
                
    pt = ("{}"
            "The parameter combination chosen by the three of you is: "
            "{}"
            "The reasons for the choices made by the other two are: "
            "{} and {}"
            "A brief summary of your reasons, as concise as possible."
            "Please select one best parameter combination from those chosen by everyone, in the format *[]*")
    keys = list(agent_prompts.keys())
    for i in range(m):
        for index, key in enumerate(agent_prompts.keys()):
            flag  = 1 #Determine whether the output meets the format.
            while flag:
                pt = pt.format(agent_prompts[key],datas,res[keys[(index+1)%3]],res[keys[(index+2)%3]])
                print(pt)
                res[i] = chat(pt)
                str = Getvalue("*",res[i])
                str = "[" + str + "]"
                try:
                    data =  ast.literal_eval(str)
                except Exception as e:
                    continue
                if len(data) == 1:
                    flag = 0
    return data[0]

#AcquisitionFunction([{"params": [0, 0], "loss": 13}], [{"params": [1, 1], "loss": 10},{"params": [2, 1], "loss": 8},{"params": [2, 2], "loss": 6},{"params": [2, 3], "loss": 0}])

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
                "As short as possible, do not provide too much information."
                "Please guess the loss for params:{} and format your output as follows: *xx*, for example *9.0* ").format(history,samples[i])
            loss = re.findall(r'-?\d+\.\d+|-?\d+', Getvalue("*",chat(pt)))
            if len(loss) == 1:
                flag = 0
                new_entry = {"params": samples[i], "loss": float(loss[0])}
                data_pred.append(new_entry)
    return data_pred



if __name__ == "__main__":
    
    ### config
    numiters = 10        # number of iters for BO
    numinipoint = 3      # number of ini points
    numsamples = 5       # number of sampled points
    
    ### initial
    inidatas = warm_strat(numiters)
    print("initial points: ",inidatas)
    datas = []
    arrloss = [1e6]
    for i in inidatas:
        loss = obj(i[0],i[1])
        new_entry = {"params": [i[0], i[1]], "loss": loss}
        arrloss[0] = min(arrloss[0],loss)
        datas.append(new_entry)
    
    ### Optimization   
    for i in tqdm(range(numiters)):
        samplers = candidate_sampling(datas,numsamples)
        data_pred = SurrogateModel(datas,samplers)
        next_point = min(data_pred, key=lambda x: x['loss'])
        loss = obj(next_point["params"][0],next_point["params"][1])
        arrloss.append(loss)
        new_entry = {"params": next_point["params"], "loss": loss} 
        datas.append(new_entry)
    
for i in range(len(arrloss)):
    arrloss[i] = min(arrloss[:i+1])
import json

data = {}
with open("BOPRO.json","r") as f:
    data = json.load(f)

data["loss"].append(arrloss)
with open("BOPRO.json","w") as f:
    json.dump(data,f)


print("final loss: ",arrloss)
    
print("The final result is: ",min(arrloss))
plt.plot(arrloss)
plt.show()
    