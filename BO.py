from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt

pbounds = {'x': (-5, 5), 'y': (-5, 5)}
def target_function(x, y):
    return -(x-2)**2 - (y - 3)**2

optimizer = BayesianOptimization(
    f=target_function,
    pbounds=pbounds,
    random_state=1,
)

# initial_points = []
# for i in range(3):
#     new_entry = {'x': inidatas[i][0], 'y': inidatas[i][1]}
#     initial_points.append(new_entry) 

# for point in initial_points:
#     optimizer.probe(
#         params=point,
#         lazy=True
#     )

# 执行优化
optimizer.maximize(
    init_points=0,
    n_iter=10,
)

target_values = [-res['target'] for res in optimizer.res]
for i in range(len(target_values)):
    target_values[i] = min(target_values[:i+1])
plt.plot(target_values, marker='o')
plt.grid(True)
plt.show()