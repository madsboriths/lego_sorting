from torchvision.models import ResNet18_Weights
import torchvision.models as models

import torch.nn as nn

def build_model(classes):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_classes = len(classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model