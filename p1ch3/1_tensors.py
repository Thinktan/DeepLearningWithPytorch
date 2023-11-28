import torch
import numpy as np

# numpy和torch.Tensor(CPU)之间互相转换

# 1. 共享底层存储

# numpy(): numpy <- tensor
a = torch.ones(5)
b = a.numpy()
a += 1
print(a, b)

# from_numpy(): tensor <- numpy
a = np.ones(5)
b = torch.from_numpy(a)
a += 2
print(a, b)

# 2. 数据拷贝
c = torch.tensor(a)
a += 3
print(a, c)