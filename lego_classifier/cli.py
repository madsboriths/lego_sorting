from pathlib import Path
import data as data
import typer

app = typer.Typer()
    
@app.command()
def train(
    data_dir: Path = typer.Option(..., "--data-dir", "-d", help="Path to data directory"),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch", "-b", help="Batch size"),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate"),):
    print("Called successfully")

    # Load data
    train_loader, test_loader, classes = data.get_dataloaders(data_dir=data_dir, batch_size=batch_size)
    print(type(train_loader))

    #Print first element in train_loader
    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        break

    # Build model

    # train epochs

    # report metrics

@app.command()
def dull():
    pass

if __name__ == "__main__":
    app()