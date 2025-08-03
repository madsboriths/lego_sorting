import torch
import os

from datetime import datetime

TRAINED_MODELS_DIR = "trained_models"

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