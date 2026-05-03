#把你之前在训练阶段（第三阶段）定义的模型代码放进这个文件里。
import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=5):
    # 随便创建一个结构（比如最简单的 ResNet18）
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
