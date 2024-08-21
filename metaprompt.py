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

def chat(says,outformat=ListResponse):
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
        "You are an experienced optimization master and adept at summarizing experiences and patterns from historical knowledge."
        "You need to summarize experiences and patterns from similar tasks, then solve new optimization problems based on these experiences and patterns."
        "Please summarize experiences and patterns based on the following tasks, histories, and knowledge."
        "tasks: {}, histories: {}, knowledge: {}"
        "Please provide your answer."
    )
    meta_prompt = chat(pt.format(json.dumps(tasks),json.dumps(historys),json.dumps(knowledge)))   
    return meta_prompt