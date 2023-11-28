import torch
from torch import hub

resnet18_model = hub.load('pytorch/vision:master', 'resnet18', pretrained=True)
print(resnet18_model)