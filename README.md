# 🖼️ Pix2Pix GAN for Image Restoration

This project implements a **Pix2Pix-style Conditional GAN** for **image restoration** on the **CelebA-HQ (256×256)** dataset.  
The goal is to train a model that restores corrupted images back to their clean/original form using a **U-Net Generator** and a **PatchGAN Discriminator**.

---

## 📌 Project Overview

Image restoration tasks (denoising, deblurring, artifact removal, etc.) are essential in computer vision.  
In this project:

- A **custom corruption pipeline** applies realistic degradations (noise, blur, compression, brightness/contrast changes) to clean images.  
- A **Pix2Pix GAN** is trained on **(corrupted → clean)** image pairs.  
- Evaluation is performed using **PSNR, SSIM, and visual comparisons**.  

---

## 📂 Dataset Preparation

### 1. Source Dataset
- **CelebA-HQ (256×256 resolution):** High-quality human face dataset.  
- Downloaded from Kaggle:  
```
/kaggle/input/celebahq-resized-256/celeba_hq_256
```

### 2. Corruption Pipeline
We artificially corrupt the clean images with different transformations:
- Gaussian Noise  
- Blur (Gaussian, Median)  
- JPEG Compression  
- Block-like artifacts  
- Brightness changes  
- Contrast changes  

Each corrupted image is paired with its original, and metadata is stored in a `.txt` file describing applied corruptions.

### 3. Dataset Organization
```
dataset/
├── train/
│ ├── original/
│ ├── corrupted/
│ └── corruption_info/
├── val/
│ ├── original/
│ ├── corrupted/
│ └── corruption_info/
└── test/
├── original/
├── corrupted/
└── corruption_info/

```
---

## 🏗️ Model Architecture

### 1. Generator — U-Net (Encoder–Decoder with Skip Connections)
- **Input shape:** `(256, 256, 3)`  
- **Encoder:** Convolutional downsampling blocks with BatchNorm, LeakyReLU, and Dropout.  
- **Bottleneck:** Deepest layer with 512 filters.  
- **Decoder:** Transposed convolutions with ReLU, Dropout, and skip connections.  
- **Output:** Restored RGB image (`tanh` activation → pixel values ∈ `[-1, 1]`).  

```python
generator = Generator(input_shape=(256, 256, 3))
```
### 2. Discriminator — PatchGAN

Input: Concatenated (corrupted_image, restored_image)

Architecture: Convolutional layers progressively downsample.

Output: Patch-wise real/fake probability map.

### 3. Combined GAN

Generator learns to restore images.

Discriminator learns to classify:

(corrupted, restored) → fake

(corrupted, original) → real

### Loss:

Adversarial Loss (BCE)

L1 Reconstruction Loss

## ⚙️ Training
### 1. Preprocessing

Normalize images to [-1, 1].

Batch size and input shape consistent at (256, 256, 3).

### 2. Training Loop

Generate restored image from corrupted input using Generator.

Train Discriminator on real pairs and fake pairs.

Train Generator via combined GAN loss:

Fool Discriminator (adversarial loss).

Stay close to ground truth (L1 loss).

### 3. Saving

Models: generator.h5, discriminator.h5, gan.h5

Sample restored outputs saved periodically during training.

# 📊 Evaluation
## Metrics

PSNR (Peak Signal-to-Noise Ratio)

SSIM (Structural Similarity Index)

(Optional) LPIPS for perceptual similarity.

## Visualization

Side-by-side comparisons:

Original image

Corrupted input

Restored output

# 🔍 Results
Example Restoration
```
Original	Corrupted	Restored
(to be added)	(to be added)	(to be added)
```
More results will be added after training.

# 🛠️ Requirements
```
Python 3.8+

TensorFlow 2.9+

NumPy, Pandas

Matplotlib, Seaborn

OpenCV, Pillow

Install dependencies:

pip install tensorflow numpy pandas matplotlib opencv-python pillow
```

# ▶️ Usage
### 1. Prepare Dataset
```
Download CelebA-HQ (256×256) and generate corrupted dataset.
```
### 2. Train the Model
```
train_pix2pix(generator, discriminator, train_loader, val_loader, epochs=200)
```
### 3. Evaluate on Test Set
```
evaluate_model(generator, test_loader)
```
### 4. Visualize Results
```
plot_comparison(original, corrupted, restored)
```
