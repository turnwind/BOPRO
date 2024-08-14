import json
import ast
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from openai import OpenAI
from xgboost import XGBClassifier
import random
seed = random.randint(0, 10000)


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

pbounds = {
    'n_estimators': (10, 200),
    'max_depth': (1, 30),
    'learning_rate': (0.01, 0.3),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
}

def obj(n_estimators, max_depth, learning_rate, subsample, colsample_bytree):
    # 创建XGBoost分类器
    model = XGBClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=seed,
        eval_metric='logloss'
    )
    
    # 使用交叉验证评估模型性能
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean()


def Getvalue(ch,str):
    s = str.find(ch)
    e = str[s+1:].find(ch)
    return str[s+1:s+e+1]

def Optimizer(history):
    flag  = 1 #Determine whether the output meets the format.
    while flag:
        pt = ("You are helping tune hyperparameters to max loss(>0)."
            "You need to address a hyperparameter optimization problem for XGBoost, utilizing the iris dataset."
            "x1(n_estimators) must be integers and within the range of [10, 200]. x2(max_depth) must be integers and within the range of [1, 30]. x3(learning_rate) must be float and within the range of [0.01, 0.3]. x4(subsample) must be float and within the range of [0.5,1.0]. x5(colsample_bytree) must be float and within the range of [0.5, 1.0]."
            "I want you to recommend a new configuration that can maximize the loss."
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


X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)


if __name__ == "__main__":
    datas = []
    
    for i in range(23):
        point = Optimizer(datas)[0]
        datas.append({"params": point, "loss": obj(point[0],point[1],point[2],point[3],point[4])})
    
    x_params = [item['params'][0] for item in datas]
    y_params = [item['params'][1] for item in datas]
    z_params = [item['params'][2] for item in datas]
    k_params = [item['params'][3] for item in datas]
    l_params = [item['params'][4] for item in datas]
    # 保存loss结果到LLM.json文件,追加写入
    data = []
    for item in datas:
        data.append(item["loss"])

    min_data = max(data[:3])
    data[2] = min_data
    # 去除前三个点
    data = data[2:]
    for i in range(len(data)):
        data[i] = max(data[:i+1])
    # 打开json文件
    with open("EXP/exp_findmin/xgboost/LLM.json", "r") as f:
        data1 = json.load(f)
        data1["loss"].append(data)
    # 写入json文件
    with open("EXP/exp_findmin/xgboost/LLM.json", "w") as f:
        json.dump(data1, f)
    
