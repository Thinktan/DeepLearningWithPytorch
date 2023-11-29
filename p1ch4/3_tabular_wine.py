import csv
import numpy as np
import torch

wine_path = '../data/p1ch4/tabular-wine/winequality-white.csv'
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=';', skiprows=1)
print(wineq_numpy)

# numpy转成torch
wineq = torch.from_numpy(wineq_numpy)
print('wineq: ', wineq.shape, wineq.dtype)

# 表示分数
data = wineq[:, :-1]
target = wineq[:, -1]
target = wineq[:, -1].long()

print('data: ', data.shape, data.dtype)
print('target: ', target.shape, target.dtype)

# print(target.unsqueeze(1))

# one-hot encoding
target_onehot = torch.zeros(target.shape[0], 10)
target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
print(target_onehot)

# 学习数据归一化
data_mean = torch.mean(data, dim=0)
data_var = torch.var(data, dim=0)

data_normalized = (data-data_mean)/torch.sqrt(data_var)
print(data_normalized)

# 选择阈值
col_list = next(csv.reader(open(wine_path), delimiter=';'))
print('col_list: \n', col_list)

bad_indexes = target <= 3
print('bad_indexes: ', bad_indexes.shape, bad_indexes.sum())

bad_data = data[bad_indexes]
print('bad_data: ', bad_data.shape)

# In[15]:
bad_data = data[target <= 3]
mid_data = data[(target > 3) & (target < 7)]
good_data = data[target >= 7]

bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)

for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))

total_sulfur_threshold = 141.83
total_sulfur_data = data[:,6]
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)

print('predicted_indexes: ', predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum())

actual_indexes = target > 5

print('actual_indexes: ', actual_indexes.shape, actual_indexes.dtype, actual_indexes.sum())

n_matches = torch.sum(actual_indexes & predicted_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()

print('ans: ', n_matches, n_matches / n_predicted, n_matches / n_actual)
