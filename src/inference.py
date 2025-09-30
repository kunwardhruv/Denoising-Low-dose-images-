import os
import sys
import torch
from PIL import Image
import torchvision.transforms as transforms

# ✅ Add parent folder to sys.path so that "src" is recognized
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.model import ResNet18_UNet

# ✅ Import metrics
from metrics import psnr, ssim, mse

# -------------------
# Config
# -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "outputs/checkpoints/best_model.pth"
TEST_DIR = "dataset/test/low"
RESULTS_DIR = "outputs/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------
# Load Model
# -------------------
model = ResNet18_UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------
# Transform (for model input only)
# -------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

to_pil = transforms.ToPILImage()

# -------------------
# Inference
# -------------------
def enhance_image(image_path, save_path):
    try:
        # Open original image
        img = Image.open(image_path).convert("L")
        orig_size = img.size  # (width, height)

        # Transform for model input (resize -> tensor -> normalize)
        inp = transform(img).unsqueeze(0).to(DEVICE)  # [1,1,256,256]

        with torch.no_grad():
            out = model(inp)

        # Denormalize back to [0,1]
        out = out.squeeze(0).cpu()  # [1,256,256]
        out = (out * 0.5 + 0.5).clamp(0, 1)

        # Convert to PIL and resize to original size
        out_img = to_pil(out).resize(orig_size, Image.BICUBIC)

        # Save enhanced image
        out_img.save(save_path)

        # -------------------
        # Metrics calculation
        # -------------------
        inp_metric = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)      # [1,1,H,W]
        out_metric = transforms.ToTensor()(out_img).unsqueeze(0).to(DEVICE) # [1,1,H,W]

        psnr_val = psnr(out_metric, inp_metric)
        ssim_val = ssim(out_metric, inp_metric)
        mse_val  = mse(out_metric, inp_metric)

        print(f"✅ Saved: {save_path} | Size: {orig_size} | PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}, MSE: {mse_val:.6f}")

    except Exception as e:
        print(f"❌ Skipping {image_path}: {e}")


def run_inference():
    for filename in os.listdir(TEST_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            in_path = os.path.join(TEST_DIR, filename)
            out_path = os.path.join(
                RESULTS_DIR,
                filename.rsplit('.', 1)[0] + "_enhanced.png"
            )
            enhance_image(in_path, out_path)


if __name__ == "__main__":
    run_inference()
