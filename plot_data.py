import json
import matplotlib.pyplot as plt
import numpy as np

data_RS = json.load(open('RS.json'))
data_BO = json.load(open('BO.json'))
data_LLM = json.load(open('LLM.json'))
data_BOPRO = json.load(open('BOPRO.json'))

data_RS = np.array(data_RS["loss"])
data_BO = np.array(data_BO["loss"])
data_LLM = np.array(data_LLM["loss"])
data_BOPRO = np.array(data_BOPRO["loss"])

# 计算每个点的平均值
average_RS = []
average_BO = []
average_LLM = []
average_BOPRO = []
for i in range(len(data_RS[0])):
    average_RS.append(np.mean(data_RS[:, i]))
for i in range(len(data_BO[0])):
    average_BO.append(np.mean(data_BO[:, i]))
for i in range(len(data_LLM[0])):
    average_LLM.append(np.mean(data_LLM[:, i]))
for i in range(len(data_BOPRO[0])):
    average_BOPRO.append(np.mean(data_BOPRO[:, i]))

# 计算每个点的标准差
std_RS = []
std_BO = []
std_LLM = []
std_BOPRO = []
for i in range(len(data_RS[0])):
    std_RS.append(np.std(data_RS[:, i]))
for i in range(len(data_BO[0])):
    std_BO.append(np.std(data_BO[:, i]))
for i in range(len(data_LLM[0])):
    std_LLM.append(np.std(data_LLM[:, i]))
for i in range(len(data_BOPRO[0])):
    std_BOPRO.append(np.std(data_BOPRO[:, i]))

# 绘制误差线
x_values = np.arange(len(data_RS[0]))
plt.plot(x_values, average_RS, label='RS')
plt.fill_between(x_values, np.array(average_RS) - np.array(std_RS), np.array(average_RS) + np.array(std_RS), alpha=0.2)
x_values = np.arange(len(data_BO[0]))
plt.plot(x_values, average_BO, label='BO')
plt.fill_between(x_values, np.array(average_BO) - np.array(std_BO), np.array(average_BO) + np.array(std_BO), alpha=0.2)
x_values = np.arange(len(data_LLM[0]))
plt.plot(x_values, average_LLM, label='LLM')
plt.fill_between(x_values, np.array(average_LLM) - np.array(std_LLM), np.array(average_LLM) + np.array(std_LLM), alpha=0.2)
x_values = np.arange(len(data_BOPRO[0]))
plt.plot(x_values, average_BOPRO, label='BOPRO')
plt.fill_between(x_values, np.array(average_BOPRO) - np.array(std_BOPRO), np.array(average_BOPRO) + np.array(std_BOPRO), alpha=0.2)
plt.legend()
plt.show()

