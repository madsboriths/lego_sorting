import json

from typing import List
from pathlib import Path

from torchvision import models
import torch.nn as nn
import torch

import lego_classifier.fileio as fileio
import lego_classifier.models as models
import lego_classifier.data_handling as data_handling

from PIL import Image

def run_inference_on_image_collection(paths: List[Path], model_path = f"{fileio.TRAINED_MODELS_DIR}/resnet_2025-03-08_17-01"):
    unpacked_image_paths = data_handling.unpack_paths(paths)
    loaded_images = fileio.load_images(unpacked_image_paths)

    with open("class_to_idx.json") as f:
        class_to_idx = json.load(f)

    classes = class_to_idx.keys()
    model = fileio.load_model(model_path, classes)
    model.eval()