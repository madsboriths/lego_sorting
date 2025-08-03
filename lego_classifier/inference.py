import json
import fileio

from torchvision import models

import torch.nn as nn
import torch

import models

import data_preparation

def run_inference(paths, model_path = f"{fileio.TRAINED_MODELS_DIR}/resnet_2025-03-08_17-01"):
    with open("class_to_idx.json") as f:
        class_to_idx = json.load(f)

    classes = class_to_idx.keys()
    model = models.load_model(model_path, classes)
    model.eval()