from pathlib import Path
import typer

from typing import List

app = typer.Typer()
    
@app.command(name="train")
def train(
    data_dir: Path = typer.Option("./data/LEGO_brick_images_v1", "--data-dir", "-d", help="Path to data directory"),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch", "-b", help="Batch size"),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate")):

    import lego_classifier.training as tr

    typer.echo(f"Training with data from {data_dir}, "
               f"for {epochs} epochs, batch size {batch_size}, learning rate {lr}")

    tr.run_training(data_dir, epochs, batch_size, lr)

@app.command(name="run-inference")
def run_inference(
    paths: List[Path] = typer.Argument(..., 
                                       exists=True, 
                                       file_okay=True, 
                                       dir_okay=True, 
                                       readable=True,
                                       resolve_path=True,
                                       help="Paths to resources")):
    # true_labels: List[Path] = typer.Option(None,"--true-labels", "-tl", help="List of true labels (optional)")
    print("Running inference on the following paths:")

    import lego_classifier.inference as inference

    paths = [Path(p) for p in paths]
    results = inference.run_inference_on_image_collection(paths)
    typer.echo(results)
    
if __name__ == "__main__":
    app()