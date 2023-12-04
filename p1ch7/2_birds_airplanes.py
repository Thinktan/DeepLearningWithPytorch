from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_printoptions(edgeitems=2)
torch.manual_seed(123)

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

from torchvision import datasets, transforms
data_path = '../data-unversioned/p1ch7/'

# 训练数据
cifar10 = datasets.CIFAR10(
    data_path, train=True, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))

# 验证数据
cifar10_val = datasets.CIFAR10(
    data_path, train=False, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))


# 提取出鸟与飞机
label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label])
          for img, label in cifar10
          if label in [0, 2]]
cifar2_val = [(img, label_map[label])
              for img, label in cifar10_val
              if label in [0, 2]]


# 构建模型
# 32*32*3 = 3072
import torch.nn as nn

n_out = 2
model = nn.Sequential(
    nn.Linear(3072, 512),
    nn.Tanh(),
    nn.Linear(512, 2),
    nn.LogSoftmax(dim=1)
)


img, _ = cifar2[0]
# plt.imshow(img.permute(1, 2, 0))
# plt.show()

#############################################
# img_batch = img.view(-1).unsqueeze(0)
# out = model(img_batch)
#
# _, index = torch.max(out, dim=1)
# print(index)
#
# loss = nn.NLLLoss()

# 训练
#
# train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64,
#                                            shuffle=True)
#
# learning_rate = 1e-2
#
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
#
# loss_fn = nn.NLLLoss()
#
# n_epochs = 100
#
# for epoch in range(n_epochs):
#     for imgs, labels in train_loader:
#         batch_size = imgs.shape[0]
#         outputs = model(imgs.view(batch_size, -1))
#         loss = loss_fn(outputs, labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
# print("Epoch: %d, Loss: %f" % (epoch, float(loss)))
#
# # 评估
# val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64,
#                                          shuffle=False)
# correct = 0
# total = 0
#
# with torch.no_grad():
#    for imgs, labels in val_loader:
#        batch_size = imgs.shape[0]
#        outputs = model(imgs.view(batch_size, -1))
#        _, predicted = torch.max(outputs, dim=1)
#        total += labels.shape[0]
#        correct += int((predicted == labels).sum())
#
# print("Accuracy: %f", correct / total)



#############################################
# 8.2 卷积实战
conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
output = conv(img.unsqueeze(0))
print('img: ', img.unsqueeze(0).shape)
print('output: ', output.shape)

# 8.2.1 用卷积检测特征

## 1 更平滑的特征图
# with torch.no_grad():
#     conv.bias.zero_()
#
# with torch.no_grad():
#     conv.weight.fill_(1.0/9.0)
#
# output = conv(img.unsqueeze(0))
# plt.imshow(output[0, 0].detach(), cmap='gray')
# plt.show()

## 2 垂直检测器
# with torch.no_grad():
#     conv.weight[:] = torch.tensor([[-1.0, 0.0, 1.0],
#                                    [-1.0, 0.0, 1.0],
#                                    [-1.0, 0.0, 1.0]])
#     conv.bias.zero_()
#
# output = conv(img.unsqueeze(0))
# plt.imshow(output[0, 0].detach(), cmap='gray')
# plt.show()


# 8.2.3 池化技术

# 1. 下采样
pool = nn.MaxPool2d(2)
output = pool(img.unsqueeze(0))
print('img: ', img.unsqueeze(0).shape)
print('output: ', output.shape)

# 8.2.4 整合网络
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.Tanh(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 8, kernel_size=3, padding=1),
    nn.Tanh(),
    nn.MaxPool2d(2),
    # missing something important 缺少从有8个通道的、8×8的图像转换为有512个元素的一维向量的步骤
    nn.Linear(8*8*8, 32),
    nn.Tanh(),
    nn.Linear(32, 2)
)

numel_list = [p.numel() for p in model.parameters()]
print(sum(numel_list), numel_list)







