from torchvision import models
import torch

# print(dir(models))

# 加载模型
resnet = models.resnet101(pretrained=True)
# print(resnet)

# 预处理图片
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

from PIL import Image
img = Image.open('../test_image_no_git/xxxx.jpg')
# img.show()
img_t = preprocess(img)
print(img_t.shape)
batch_t = torch.unsqueeze(img_t, 0)
print(batch_t.shape)

# 推理
resnet.eval()
out = resnet(batch_t)
print(out.shape)

# 展示结果
# with open('../data/p1ch2/imagenet_classes.txt') as f:
with open('../data/p1ch2/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

# print(labels)

_, index = torch.max(out, 1)
# print(labels[index[0]])

percentage = torch.nn.functional.softmax(out, dim=1)[0]*100
print(labels[index[0]], percentage[index[0]].item())

_, indices = torch.sort(out, descending=True)
print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])


