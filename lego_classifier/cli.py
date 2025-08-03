from pathlib import Path
import data as data
import typer

import torch
import training as tr

app = typer.Typer()
    
@app.command()
def train(
    data_dir: Path = typer.Option("./data/LEGO_brick_images_v1", "--data-dir", "-d", help="Path to data directory"),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch", "-b", help="Batch size"),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate")):

    print(f"Training with data from {data_dir}, "
          f"for {epochs} epochs, batch size {batch_size}, learning rate {lr}")

    tr.run_training(data_dir, epochs, batch_size, lr)

@app.command()
def dull():
    pass

if __name__ == "__main__":
    app()