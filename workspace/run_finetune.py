import argparse
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import NewDataset
from models import MyOpticalFlowModel
from multiscaleloss import multiscaleEPE
from util import save_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Optical Flow Model")
    parser.add_argument("data", metavar="DIR", help="path to fine-tuning dataset")
    parser.add_argument("--pretrained", required=True, help="path to pre-trained model")
    parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
    parser.add_argument("--batch-size", default=4, type=int, help="batch size")
    parser.add_argument("--epochs", default=50, type=int, help="number of epochs")
    parser.add_argument("--save-path", default="./fine_tuned", help="path to save model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255])
    ])
    train_dataset = NewDataset(args.data, split="train", transform=transform)
    val_dataset = NewDataset(args.data, split="val", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model Loading
    print("=> Loading pre-trained model from {}".format(args.pretrained))
    model = MyOpticalFlowModel()
    checkpoint = torch.load(args.pretrained)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    # Optimizer and Scheduler
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Fine-tuning Loop
    best_loss = float("inf")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss = train_epoch(train_loader, model, optimizer, device)
        val_loss = validate_epoch(val_loader, model, device)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step()

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint({"state_dict": model.state_dict()}, True, args.save_path)

def train_epoch(loader, model, optimizer, device):
    model.train()
    running_loss = 0
    for input, target in loader:
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = multiscaleEPE(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate_epoch(loader, model, device):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for input, target in loader:
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = multiscaleEPE(output, target)
            running_loss += loss.item()
    return running_loss / len(loader)

if __name__ == "__main__":
    main()
