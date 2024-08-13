from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

# 设置随机数种子
import random
seed = random.randint(0, 10000)




inidatas = [[3,4], [1, 4], [0.9,3.9]]
pbounds = {'x': (-5, 5), 'y': (-5, 5)}
def target_function(x, y):
    return -(x-2)**2 - (y - 3)**2

optimizer = BayesianOptimization(
    f=target_function,
    pbounds=pbounds,
    random_state=seed,
)
# 执行优化
optimizer.maximize(
    init_points=3,
    n_iter=11,
)

target_values = [-res['target'] for res in optimizer.res]
for i in range(len(target_values)):
    target_values[i] = min(target_values[:i+1])

# 将后10个点存入json BO.json,追加
import json
data = json.load(open('BO.json'))

data["loss"] .append(target_values[-10:])
# save
with open('BO.json', 'w') as f:
    json.dump(data, f)