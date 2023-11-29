import torch

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
print(points.storage())

points_storage = points.storage()
print(points_storage[0])
print(points.storage()[1])

points_storage[0] = 2.0
print(points)

# 就地操作
a = torch.ones(3, 2)
print("a: ", a)
a.zero_()
print("a: ", a)

# 转置
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_t = points.t()
print('points: ', points)
print('points_t: ', points_t)
print(id(points.storage()) == id(points_t.storage()))
print('points.stride(): ', points.stride())
print('points_t.stride(): ', points_t.stride())










