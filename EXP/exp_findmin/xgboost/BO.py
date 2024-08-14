from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import random
import json
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

seed = random.randint(0, 10000)

# 定义超参数搜索范围
pbounds = {
    'n_estimators': (10, 200),
    'max_depth': (1, 30),
    'learning_rate': (0.01, 0.3),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
}
def target_function(n_estimators, max_depth, learning_rate, subsample, colsample_bytree):
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
# 生成示例数据集
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# 创建贝叶斯优化器实例
optimizer = BayesianOptimization(
    f=target_function,
    pbounds=pbounds,
    random_state=seed,
)

# 执行优化
optimizer.maximize(
    init_points=3,  # 初始随机探索次数
    n_iter=20,      # 优化迭代次数
)

# 获取最佳参数和目标值
best_params = optimizer.max['params']
best_score = optimizer.max['target']

# 输出最佳参数和得分
print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation Score: {best_score}")

# 保存结果
target_values = [res['target'] for res in optimizer.res]
for i in range(len(target_values)):
    target_values[i] = max(target_values[:i+1])

try:
    data = json.load(open('EXP/exp_findmin/xgboost/BO.json'))
except FileNotFoundError:
    data = {"loss": []}

data["loss"].append(target_values[-21:])

with open('EXP/exp_findmin/xgboost/BO.json', 'w') as f:
    json.dump(data, f)

# 可视化优化过程
# plt.plot(target_values)
# plt.xlabel('Iteration')
# plt.ylabel('Best Loss (Negative Accuracy)')
# plt.title('Bayesian Optimization of Random Forest')
# plt.show()
