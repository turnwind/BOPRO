import random
import json
import matplotlib.pyplot as plt

seed = random.randint(0, 10000)
random.seed(seed)

pbounds = {'x': (-5, 5), 'y': (-5, 5), 'z': (-5, 5)}
def target_function(x, y, z):
    return -(x-1.5)**2 - (y - 0.5)**2 - (z + 1)**2

class RandomSearchOptimizer:
    def __init__(self, target_function, pbounds, random_state=None):
        self.target_function = target_function
        self.pbounds = pbounds
        self.random_state = random_state
        self.res = []

    def maximize(self, init_points=3, n_iter=10):
        for _ in range(init_points + n_iter):
            x = random.uniform(self.pbounds['x'][0], self.pbounds['x'][1])
            y = random.uniform(self.pbounds['y'][0], self.pbounds['y'][1])
            z = random.uniform(self.pbounds['z'][0], self.pbounds['z'][1])
            target = self.target_function(x, y,z)
            self.res.append({'target': target, 'params': {'x': x, 'y': y, 'z': z}})
        # self.res.sort(key=lambda x: x['target'], reverse=True)

optimizer = RandomSearchOptimizer(
    target_function=target_function,
    pbounds=pbounds,
    random_state=seed,
)

optimizer.maximize(
    init_points=3,
    n_iter=11,
)


target_values = [-res['target'] for res in optimizer.res]
for i in range(len(target_values)):
    target_values[i] = min(target_values[:i+1])

try:
    data = json.load(open('EXP/exp_findmin/cubic_function/RS.json'))
except FileNotFoundError:
    data = {"loss": []}

data["loss"].append(target_values[-11:])

with open('EXP/exp_findmin/cubic_function/RS.json', 'w') as f:
    json.dump(data, f)

# 可视化
# plt.plot(target_values)
# plt.xlabel('Iteration')
# plt.ylabel('Best Loss')
# plt.title('Random Search Optimization')
# plt.show()
