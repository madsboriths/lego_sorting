import torch

import lego_classifier.model_engine as model_engine
import lego_classifier.data_handling as data_handling
import lego_classifier.fileio as fileio
import lego_classifier.models as models

import torch.nn as nn

def run_training(data_dir, epochs, batch_size, lr):
    """
    parameters:
      data_dir (str): training data directory. Subfolder names are assumed to be classnames.
    """
    dataset_ImageFolder = data_handling.make_dataset(data_dir)
    train_loader, test_loader = data_handling.get_dataloaders_from_ImageFolder(dataset_ImageFolder)

    model = models.build_model(dataset_ImageFolder.classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float("inf")
    # model_engine.training_loop()
    for epoch in range(1, epochs+1):
        train_loss, train_acc = model_engine.train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = model_engine.validate_one_epoch(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch:2d} | "
          f"Train loss {train_loss:.4f}, acc {train_acc:.4f} | "
          f"Val loss {val_loss:.4f}, acc {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            model_name = model.__class__.__name__.lower()
            run_dir = fileio.make_run_dir(model_name)
            fileio.save_best_model(model, run_dir)
            