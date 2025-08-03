from torchvision.models import ResNet18_Weights
import torchvision.models as models

import torch
import torch.nn as nn

import fileio

def build_model(classes):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_classes = len(classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def load_model(model_path, classes):
    file_path = f"{model_path}/{fileio.BEST_MODEL_FILE_NAME}"
    model = build_model(classes)
    model.load_state_dict(torch.load(file_path))
    return model