import torch
import os

from datetime import datetime

from pathlib import Path
from typing import List

from PIL import Image

import lego_classifier.data_handling as data_handling
import lego_classifier.models as models

TRAINED_MODELS_DIR = "trained_models"
BEST_MODEL_FILE_NAME = "best.pth"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def make_run_dir(model_name: str) -> str:
    ts = datetime.now().strftime("%Y-%d-%m_%H-%M")
    run_name = f"{model_name}_{ts}"
    run_dir = os.path.join(TRAINED_MODELS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_best_model(model, run_dir: str):
    path = os.path.join(run_dir, "best.pth")
    torch.save(model.state_dict(), path)
    return path

def load_model(model_path, classes):
    file_path = f"{model_path}/{BEST_MODEL_FILE_NAME}"
    model = models.build_model(classes)
    model.load_state_dict(torch.load(file_path))
    return model

def load_images(paths: List[Path]) -> List[Image.Image]:
    return [Image.open(p).convert("RGB") for p in paths]