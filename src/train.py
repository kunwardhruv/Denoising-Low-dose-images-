import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from src.dataloader import get_dataloaders
from src.model import ResNet18_UNet
 
# -------------------
# Config
# -------------------
DATA_ROOT = "dataset"
EPOCHS = 20
BATCH_SIZE = 4
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "outputs/checkpoints/best_model.pth"
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)  # auto create folder

# -------------------
# Training Function
# -------------------
def train(): 
    # Load data
    train_loader, val_loader, _ = get_dataloaders(DATA_ROOT, batch_size=BATCH_SIZE)

    # Model
    model = ResNet18_UNet().to(DEVICE)

    # Loss & Optimizer
    criterion = nn.L1Loss()  # pixel-wise difference
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # ---- Training ----
        model.train()
        train_loss = 0
        for low, high in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            low, high = low.to(DEVICE), high.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(low)
            
            # Resize outputs to match high-dose image size
            outputs = F.interpolate(outputs, size=high.shape[2:], mode='bilinear', align_corners=False)

            loss = criterion(outputs, high)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for low, high in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                low, high = low.to(DEVICE), high.to(DEVICE)
                outputs = model(low)
                
                # Resize outputs to match high-dose image size
                outputs = F.interpolate(outputs, size=high.shape[2:], mode='bilinear', align_corners=False)

                loss = criterion(outputs, high)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"✅ Best model saved at {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
