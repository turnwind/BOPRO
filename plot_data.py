import json
import matplotlib.pyplot as plt
import numpy as np

# Load the data BO_data.json
data = json.load(open('RS.json'))
data = np.array(data["loss"])

# Plot the data
x_values = np.arange( len(data[0]))

# 计算每个点的平均值
average = []
for i in range(len(data[0])):
    average.append(np.mean(data[:, i]))

# 计算每个点的标准差
std = []
for i in range( len(data[0])):
    std.append(np.std(data[:, i]))
# 绘制误差线
print(std)
print(average)
print(x_values)


print(std)
plt.errorbar(x_values, average, yerr=std, fmt='-o', label='BO')
plt.fill_between(x_values, np.array(average) - np.array(std), np.array(average) + np.array(std), alpha=0.2)
plt.show()
