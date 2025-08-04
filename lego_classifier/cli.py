from pathlib import Path
import typer

import lego_classifier.training as tr
import lego_classifier.data_handling as data_handling
import lego_classifier.inference as inference

from typing import List

app = typer.Typer()
    
@app.command(name="train")
def train(
    data_dir: Path = typer.Option("./data/LEGO_brick_images_v1", "--data-dir", "-d", help="Path to data directory"),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch", "-b", help="Batch size"),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate")):

    typer.echo(f"Training with data from {data_dir}, "
               f"for {epochs} epochs, batch size {batch_size}, learning rate {lr}")

    tr.run_training(data_dir, epochs, batch_size, lr)

@app.command(name="run-inference")
def run_inference(
    paths: List[Path] = typer.Argument(..., help="Paths to resources")):

    typer.echo(f"Running inference on {len(paths)} images")
    results = inference.run_inference_on_image_collection(paths)
    
if __name__ == "__main__":
    app()