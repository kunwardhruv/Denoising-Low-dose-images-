# 🩻 CNN-Based Low-Dose X-Ray Image Denoising


# 🚀 Project Overview

This project focuses on denoising and enhancing low-dose X-ray images using a deep learning-based CNN model built with ResNet-18 + UNet architecture.

In medical imaging, radiation exposure is harmful to the human body.
So, instead of capturing high-dose (clear) X-rays, we capture low-dose (safer but noisy) X-rays and then enhance them using our trained CNN model.

Our model restores the low-dose X-rays to near high-dose quality — making them clear, detailed, and diagnostically useful for doctors while reducing radiation risk to patients.

---

# 🧠 Objective

To denoise and enhance low-dose X-ray images using CNN architecture, improving image clarity while minimizing radiation exposure to patients.

---

# 📁 Folder Structure

```
PRE-TRAINED CNN/
│
├── dataset/
│   ├── train/
│   │   ├── high/
│   │   └── low/
│   ├── val/
│   │   ├── high/
│   │   └── low/
│   └── test/
│       ├── high/
│       └── low/
│
├── outputs/
│   ├── checkpoints/
│   │   └── best_model.pth
│   └── results/
│
├── src/
│   ├── dataloader.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   └── metrics.py
│
├── cnn_code.md / txt
└── README.md
```
---


# 🧩 Model Architecture

The model combines:

ResNet-18 Encoder: extracts deep semantic features from noisy X-ray images.

UNet Decoder: reconstructs the enhanced, denoised image from encoded features.

This hybrid approach allows the model to retain important structural details while effectively removing noise.

# ⚙️ How It Works

Low-dose X-ray is captured with minimal radiation exposure.

The CNN (ResNet18 + UNet) model processes this noisy image.

The output is a clean, enhanced X-ray image with high clarity.


# 🖼️ Results


🔹 Before Enhancement (Low-Dose Input)

<img width="275" height="277" alt="Screenshot (236)" src="https://github.com/user-attachments/assets/ab3179ec-e622-4099-800f-e2aa5b1d76ee" />

---

🔹 After Enhancement (Model Output)

<img width="449" height="509" alt="Screenshot (235)" src="https://github.com/user-attachments/assets/3c81d8a0-ad36-473f-85ec-7a098196a064" />



The enhanced X-ray clearly shows anatomical structures with reduced noise, allowing doctors to detect diseases effectively without high radiation exposure.
---

 ## 🧪 Training Details

Architecture: ResNet18 + UNet

Framework: PyTorch

Loss Function: MSELoss / L1Loss (for pixel-wise denoising)

Optimizer: Adam

Dataset: Custom low-dose and high-dose X-ray dataset

Output: Best model saved as outputs/checkpoints/best_model.pth

## 📊 Evaluation Metrics

The model performance is evaluated using:

PSNR (Peak Signal-to-Noise Ratio)

SSIM (Structural Similarity Index)

MSE (Mean Squared Error)

Higher PSNR and SSIM values indicate better denoising performance.
---

# ▶️ How to Run

```
1.) Clone the repository

git clone https://github.com/<your-username>/pre-trained-cnn.git
cd pre-trained-cnn


2.) Install dependencies

pip install -r requirements.txt


3.) Train the model

python src/train.py


4.) Run inference

python src/inference.py

📦 Output Example

The trained model generates denoised X-rays and saves them in:

outputs/results/

```
---

# 🧑‍⚕️ Impact

This model contributes to safer radiology practices by:

Reducing harmful radiation exposure.

Maintaining diagnostic quality of X-rays.

Helping doctors analyze clearer images without compromising patient safety.

---

| Component    | Tool                         |
| ------------ | ---------------------------- |
| Framework    | PyTorch                      |
| Language     | Python                       |
| Model        | ResNet-18 + UNet             |
| Environment  | VS Code                      |
| Dataset Type | Low-Dose vs High-Dose X-rays |
