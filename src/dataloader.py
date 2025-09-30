import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class XrayDataset(Dataset):
    def __init__(self, low_dir, high_dir, transform=None):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.low_images = sorted(os.listdir(low_dir))
        self.high_images = sorted(os.listdir(high_dir))
        self.transform = transform

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_path = os.path.join(self.low_dir, self.low_images[idx])
        high_path = os.path.join(self.high_dir, self.high_images[idx])

        low_img = Image.open(low_path).convert("L")   # Grayscale
        high_img = Image.open(high_path).convert("L")

        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)

        return low_img, high_img


def get_dataloaders(data_root, batch_size=4):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale
    ])

    train_dataset = XrayDataset(
        os.path.join(data_root, "train/low"),
        os.path.join(data_root, "train/high"),
        transform=transform
    )

    val_dataset = XrayDataset(
        os.path.join(data_root, "val/low"),
        os.path.join(data_root, "val/high"),
        transform=transform
    )

    test_dataset = XrayDataset(
        os.path.join(data_root, "test/low"),
        os.path.join(data_root, "test/high"),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
