#### TRAIN.PY







import os

import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from tqdm import tqdm

from src.dataloader import get\_dataloaders

from src.model import ResNet18\_UNet



\# -------------------

\# Config

\# -------------------

DATA\_ROOT = "dataset"

EPOCHS = 20

BATCH\_SIZE = 4

LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is\_available() else "cpu")



CHECKPOINT\_PATH = "outputs/checkpoints/best\_model.pth"

os.makedirs(os.path.dirname(CHECKPOINT\_PATH), exist\_ok=True)  # auto create folder



\# -------------------

\# Training Function

\# -------------------

def train():

&nbsp;   # Load data

&nbsp;   train\_loader, val\_loader, \_ = get\_dataloaders(DATA\_ROOT, batch\_size=BATCH\_SIZE)



&nbsp;   # Model

&nbsp;   model = ResNet18\_UNet().to(DEVICE)



&nbsp;   # Loss \& Optimizer

&nbsp;   criterion = nn.L1Loss()  # pixel-wise difference

&nbsp;   optimizer = optim.Adam(model.parameters(), lr=LR)



&nbsp;   best\_val\_loss = float("inf")



&nbsp;   for epoch in range(EPOCHS):

&nbsp;       # ---- Training ----

&nbsp;       model.train()

&nbsp;       train\_loss = 0

&nbsp;       for low, high in tqdm(train\_loader, desc=f"Epoch {epoch+1}/{EPOCHS} \[Train]"):

&nbsp;           low, high = low.to(DEVICE), high.to(DEVICE)



&nbsp;           optimizer.zero\_grad()

&nbsp;           outputs = model(low)

&nbsp;           

&nbsp;           # Resize outputs to match high-dose image size

&nbsp;           outputs = F.interpolate(outputs, size=high.shape\[2:], mode='bilinear', align\_corners=False)



&nbsp;           loss = criterion(outputs, high)

&nbsp;           loss.backward()

&nbsp;           optimizer.step()



&nbsp;           train\_loss += loss.item()



&nbsp;       avg\_train\_loss = train\_loss / len(train\_loader)



&nbsp;       # ---- Validation ----

&nbsp;       model.eval()

&nbsp;       val\_loss = 0

&nbsp;       with torch.no\_grad():

&nbsp;           for low, high in tqdm(val\_loader, desc=f"Epoch {epoch+1}/{EPOCHS} \[Val]"):

&nbsp;               low, high = low.to(DEVICE), high.to(DEVICE)

&nbsp;               outputs = model(low)

&nbsp;               

&nbsp;               # Resize outputs to match high-dose image size

&nbsp;               outputs = F.interpolate(outputs, size=high.shape\[2:], mode='bilinear', align\_corners=False)



&nbsp;               loss = criterion(outputs, high)

&nbsp;               val\_loss += loss.item()



&nbsp;       avg\_val\_loss = val\_loss / len(val\_loader)



&nbsp;       print(f"Epoch \[{epoch+1}/{EPOCHS}] Train Loss: {avg\_train\_loss:.4f}, Val Loss: {avg\_val\_loss:.4f}")



&nbsp;       # Save best model

&nbsp;       if avg\_val\_loss < best\_val\_loss:

&nbsp;           best\_val\_loss = avg\_val\_loss

&nbsp;           torch.save(model.state\_dict(), CHECKPOINT\_PATH)

&nbsp;           print(f"✅ Best model saved at {CHECKPOINT\_PATH}")





if \_\_name\_\_ == "\_\_main\_\_":

&nbsp;   train()









#### INFERENCE.PY









import os

import sys

import torch

from PIL import Image

import torchvision.transforms as transforms



\# ✅ Add parent folder to sys.path so that "src" is recognized

sys.path.append(os.path.join(os.path.dirname(\_\_file\_\_), ".."))

from src.model import ResNet18\_UNet



\# -------------------

\# Config

\# -------------------

DEVICE = torch.device("cuda" if torch.cuda.is\_available() else "cpu")

MODEL\_PATH = "outputs/checkpoints/best\_model.pth"

TEST\_DIR = "dataset/test/low"

RESULTS\_DIR = "outputs/results"



os.makedirs(RESULTS\_DIR, exist\_ok=True)



\# -------------------

\# Load Model

\# -------------------

model = ResNet18\_UNet().to(DEVICE)

model.load\_state\_dict(torch.load(MODEL\_PATH, map\_location=DEVICE))

model.eval()



\# -------------------

\# Transform (for model input only)

\# -------------------

transform = transforms.Compose(\[

&nbsp;   transforms.Resize((256, 256)),

&nbsp;   transforms.ToTensor(),

&nbsp;   transforms.Normalize((0.5,), (0.5,))

])



to\_pil = transforms.ToPILImage()





\# -------------------

\# Inference

\# -------------------

def enhance\_image(image\_path, save\_path):

&nbsp;   try:

&nbsp;       # Open original image

&nbsp;       img = Image.open(image\_path).convert("L")

&nbsp;       orig\_size = img.size  # (width, height)



&nbsp;       # Transform for model input (resize -> tensor -> normalize)

&nbsp;       inp = transform(img).unsqueeze(0).to(DEVICE)  # \[1,1,256,256]



&nbsp;       with torch.no\_grad():

&nbsp;           out = model(inp)



&nbsp;       # Denormalize back to \[0,1]

&nbsp;       out = out.squeeze(0).cpu()  # \[1,256,256]

&nbsp;       out = (out \* 0.5 + 0.5).clamp(0, 1)



&nbsp;       # Convert to PIL

&nbsp;       out\_img = to\_pil(out)



&nbsp;       # ✅ Resize back to original size

&nbsp;       out\_img = out\_img.resize(orig\_size, Image.BICUBIC)



&nbsp;       # Save

&nbsp;       out\_img.save(save\_path)

&nbsp;       print(f"✅ Saved: {save\_path} | Size: {orig\_size}")



&nbsp;   except Exception as e:

&nbsp;       print(f"❌ Skipping {image\_path}: {e}")





def run\_inference():

&nbsp;   for filename in os.listdir(TEST\_DIR):

&nbsp;       if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):

&nbsp;           in\_path = os.path.join(TEST\_DIR, filename)

&nbsp;           out\_path = os.path.join(

&nbsp;               RESULTS\_DIR,

&nbsp;               filename.rsplit('.', 1)\[0] + "\_enhanced.png"

&nbsp;           )

&nbsp;           enhance\_image(in\_path, out\_path)





if \_\_name\_\_ == "\_\_main\_\_":

&nbsp;   run\_inference()















#### MODEL 











import torch

import torch.nn as nn

import torchvision.models as models



class EncoderResNet18(nn.Module):

&nbsp;   def \_\_init\_\_(self):

&nbsp;       super().\_\_init\_\_()

&nbsp;       resnet = models.resnet18(pretrained=True)



&nbsp;       # Change first conv layer to accept 1 channel (grayscale)

&nbsp;       self.enc1 = nn.Sequential(

&nbsp;           nn.Conv2d(1, 64, kernel\_size=7, stride=2, padding=3, bias=False),

&nbsp;           resnet.bn1,

&nbsp;           resnet.relu

&nbsp;       )

&nbsp;       self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)  # 64

&nbsp;       self.enc3 = resnet.layer2                                   # 128

&nbsp;       self.enc4 = resnet.layer3                                   # 256

&nbsp;       self.enc5 = resnet.layer4                                   # 512



&nbsp;   def forward(self, x):

&nbsp;       x1 = self.enc1(x)  # \[B,64,H/2,W/2]

&nbsp;       x2 = self.enc2(x1) # \[B,64,H/4,W/4]

&nbsp;       x3 = self.enc3(x2) # \[B,128,H/8,W/8]

&nbsp;       x4 = self.enc4(x3) # \[B,256,H/16,W/16]

&nbsp;       x5 = self.enc5(x4) # \[B,512,H/32,W/32]

&nbsp;       return x1, x2, x3, x4, x5





class DecoderBlock(nn.Module):

&nbsp;   def \_\_init\_\_(self, in\_channels, out\_channels):

&nbsp;       super().\_\_init\_\_()

&nbsp;       self.block = nn.Sequential(

&nbsp;           nn.ConvTranspose2d(in\_channels, out\_channels, kernel\_size=2, stride=2),

&nbsp;           nn.ReLU(inplace=True),

&nbsp;           nn.Conv2d(out\_channels, out\_channels, kernel\_size=3, padding=1),

&nbsp;           nn.ReLU(inplace=True)

&nbsp;       )



&nbsp;   def forward(self, x):

&nbsp;       return self.block(x)





class ResNet18\_UNet(nn.Module):

&nbsp;   def \_\_init\_\_(self):

&nbsp;       super().\_\_init\_\_()

&nbsp;       self.encoder = EncoderResNet18()



&nbsp;       self.dec5 = DecoderBlock(512, 256)

&nbsp;       self.dec4 = DecoderBlock(512, 128)  # skip connection

&nbsp;       self.dec3 = DecoderBlock(256, 64)

&nbsp;       self.dec2 = DecoderBlock(128, 64)

&nbsp;       self.dec1 = nn.Conv2d(128, 1, kernel\_size=1)  # grayscale output



&nbsp;   def forward(self, x):

&nbsp;       x1, x2, x3, x4, x5 = self.encoder(x)



&nbsp;       d5 = self.dec5(x5)

&nbsp;       d4 = self.dec4(torch.cat(\[d5, x4], dim=1))

&nbsp;       d3 = self.dec3(torch.cat(\[d4, x3], dim=1))

&nbsp;       d2 = self.dec2(torch.cat(\[d3, x2], dim=1))

&nbsp;       d1 = self.dec1(torch.cat(\[d2, x1], dim=1))



&nbsp;       return torch.tanh(d1)  # output \[-1,1] matching normalized grayscale







































#### DATASET LOADER 







import os

from torch.utils.data import Dataset, DataLoader

from PIL import Image

import torchvision.transforms as transforms



class XrayDataset(Dataset):

&nbsp;   def \_\_init\_\_(self, low\_dir, high\_dir, transform=None):

&nbsp;       self.low\_dir = low\_dir

&nbsp;       self.high\_dir = high\_dir

&nbsp;       self.low\_images = sorted(os.listdir(low\_dir))

&nbsp;       self.high\_images = sorted(os.listdir(high\_dir))

&nbsp;       self.transform = transform



&nbsp;   def \_\_len\_\_(self):

&nbsp;       return len(self.low\_images)



&nbsp;   def \_\_getitem\_\_(self, idx):

&nbsp;       low\_path = os.path.join(self.low\_dir, self.low\_images\[idx])

&nbsp;       high\_path = os.path.join(self.high\_dir, self.high\_images\[idx])



&nbsp;       low\_img = Image.open(low\_path).convert("L")   # Grayscale

&nbsp;       high\_img = Image.open(high\_path).convert("L")



&nbsp;       if self.transform:

&nbsp;           low\_img = self.transform(low\_img)

&nbsp;           high\_img = self.transform(high\_img)



&nbsp;       return low\_img, high\_img





def get\_dataloaders(data\_root, batch\_size=4):

&nbsp;   transform = transforms.Compose(\[

&nbsp;       transforms.Resize((256, 256)),

&nbsp;       transforms.ToTensor(),  

&nbsp;       transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale

&nbsp;   ])



&nbsp;   train\_dataset = XrayDataset(

&nbsp;       os.path.join(data\_root, "train/low"),

&nbsp;       os.path.join(data\_root, "train/high"),

&nbsp;       transform=transform

&nbsp;   )



&nbsp;   val\_dataset = XrayDataset(

&nbsp;       os.path.join(data\_root, "val/low"),

&nbsp;       os.path.join(data\_root, "val/high"),

&nbsp;       transform=transform

&nbsp;   )



&nbsp;   test\_dataset = XrayDataset(

&nbsp;       os.path.join(data\_root, "test/low"),

&nbsp;       os.path.join(data\_root, "test/high"),

&nbsp;       transform=transform

&nbsp;   )



&nbsp;   train\_loader = DataLoader(train\_dataset, batch\_size=batch\_size, shuffle=True)

&nbsp;   val\_loader = DataLoader(val\_dataset, batch\_size=batch\_size, shuffle=False)

&nbsp;   test\_loader = DataLoader(test\_dataset, batch\_size=1, shuffle=False)



&nbsp;   return train\_loader, val\_loader, test\_loader



