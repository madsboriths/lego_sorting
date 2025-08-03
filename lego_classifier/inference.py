import json

from torchvision import models

import torch.nn as nn
import torch

import models

def run_inference():
    with open("class_to_idx.json") as f:
        class_to_idx = json.load(f)
    
    model = models.build_model()
    model.load_state_dict(torch.load('best_resnet18.pth'))
    model.eval()

    print(type(class_to_idx))
    # idx_to_class = {v: k for k, v in class_to_idx.items()}