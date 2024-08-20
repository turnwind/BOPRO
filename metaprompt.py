from openai import OpenAI
from pydantic import BaseModel
import json 

class Step(BaseModel):
    explanation: str
    output: str

class ListResponse(BaseModel):
    steps: list[Step]
    final_answer: str

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
 
def create_metaprompt(tasks,historys,knowledge):
    pt = (
        " 你是一个经验丰富的优化大师并且善于从历史经验和知识中总结经验和规律。"
        "你需要从相似的任务中总结经验和规律，然后根据这些经验和规律来解决新的优化问题。"
        "请根据以下任务、历史和知识总结经验和规律。"
        "tasks: {}, historys: {}, knowledge: {}"
        "请给出你的答案。"
    )
    meta_prompt = chat(pt.format(json.dumps(tasks),json.dumps(historys),json.dumps(knowledge)))   
    return meta_prompt