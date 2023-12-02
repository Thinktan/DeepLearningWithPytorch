
from matplotlib import pyplot as plt
import numpy as np
import torch

torch.set_printoptions(edgeitems=2, linewidth=75)
torch.manual_seed(123)

from torchvision import datasets
data_path = '../data-unversioned/p1ch7/'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

fig = plt.figure(figsize=(8,3))
num_classes = 10
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    ax.set_title(class_names[i])
    img = next(img for img, label in cifar10 if label == i)
    plt.imshow(img)
plt.show()

print(type(cifar10).__mro__)

img, label = cifar10[99]
print(img, label, class_names[label])

plt.imshow(img)
plt.show()

from torchvision import transforms
dir(transforms)


# 数据正规化
tensor_cifar10 = datasets.CIFAR10(data_path, train=True, download=False,
                          transform=transforms.ToTensor())

imgs = torch.stack([img_t for img_t, _ in tensor_cifar10], dim=3)
print(imgs.shape) # 3, 24, 24, 50000

# permute用法： https://blog.csdn.net/york1996/article/details/81876886
# view用法： https://blog.csdn.net/york1996/article/details/81949843

# 计算平均值 3,(32*32*50000)
mean = imgs.view(3, -1).mean(dim=1)   #view(3,−1)保留了3个通道，并将剩余的所有维度合并为一个维度，从而计算出适当的尺寸大小。这里我们的3×32×32的图像被转换成一个3×1024的向量，然后对每个通道的1024个元素取平均值
# (0.4915, 0.4823, 0.4468)

# 计算标准差
std = imgs.view(3, -1).std(dim=1)
# (0.2470, 0.2435, 0.2616)

# 构造转换器
transformed_cifar10 = datasets.CIFAR10(
    data_path, train=True, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                         (0.2470, 0.2435, 0.2616))
    ]))


