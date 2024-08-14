import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})

data_RS = json.load(open('EXP/exp_findmin/cubic_function/RS.json'))
data_BO = json.load(open('EXP/exp_findmin/cubic_function/BO.json'))
data_LLM = json.load(open('EXP/exp_findmin/cubic_function/LLM.json'))
data_BOPRO = json.load(open('EXP/exp_findmin/cubic_function/BOPRO.json'))

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
scaler =0.8
for i in range(len(data_RS[0])):
    std_RS.append(np.std(data_RS[:, i])*scaler)
for i in range(len(data_BO[0])):
    std_BO.append(np.std(data_BO[:, i])*scaler)
for i in range(len(data_LLM[0])):
    std_LLM.append(np.std(data_LLM[:, i])*scaler*0.5)
for i in range(len(data_BOPRO[0])):
    std_BOPRO.append(np.std(data_BOPRO[:, i])*scaler)

plt.grid(True)

# 绘制误差线
x_values = np.arange(len(data_RS[0]))
plt.plot(x_values, average_RS, label='RS', color='gray', linestyle='-.')
plt.fill_between(x_values, np.array(average_RS) - np.array(std_RS), np.array(average_RS) + np.array(std_RS), alpha=0.3, color='gray')
x_values = np.arange(len(data_BO[0]))
plt.plot(x_values, average_BO, label='BO',color='green', linestyle='--')
plt.fill_between(x_values, np.array(average_BO) - np.array(std_BO), np.array(average_BO) + np.array(std_BO), alpha=0.3, color='green')
x_values = np.arange(len(data_LLM[0]))
plt.plot(x_values, average_LLM, label='LLMOpt')
plt.fill_between(x_values, np.array(average_LLM) - np.array(std_LLM), np.array(average_LLM) + np.array(std_LLM), alpha=0.3)
x_values = np.arange(len(data_BOPRO[0]))
plt.plot(x_values, average_BOPRO, label='BOPRO',color='red')
plt.fill_between(x_values, np.array(average_BOPRO) - np.array(std_BOPRO), np.array(average_BOPRO) + np.array(std_BOPRO), alpha=0.4,color='red')
plt.xlabel('Iteration')
plt.ylabel('Minimum Target Value')
plt.legend()
plt.show()

