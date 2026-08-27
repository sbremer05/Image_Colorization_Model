# Conditional GAN Image Colorization

A PyTorch-based conditional GAN for automatic image colorization, using a U-Net encoder-decoder generator and PatchGAN discriminator to reconstruct realistic color images from grayscale inputs.

The project uses the COCO 2017 dataset and combines supervised reconstruction with adversarial training to improve color fidelity and structural consistency.

## Overview

This project frames image colorization as a supervised conditional generation problem.

Given a grayscale image, the model predicts the missing color channels:

    Grayscale Image
           │
           ▼
     ResNet-34 Encoder
           │
           ▼
       U-Net Decoder
           │
           ▼
      Predicted A/B
      Color Channels
           │
           ▼
       Colorized Image

Images are converted from RGB to LAB color space, with the L channel used as the model input and the A/B channels used as the prediction target.

## Model

### Generator

The generator combines a pretrained ResNet-34 encoder with a U-Net-style decoder.

- ResNet-34 extracts hierarchical image features
- U-Net skip connections preserve spatial information
- Transposed convolutions progressively reconstruct the image
- The final layer predicts the A/B color channels

The grayscale L channel is replicated to three channels before being passed through the pretrained ResNet-34 encoder.

### Discriminator

A PatchGAN discriminator is used to distinguish between real and generated colorizations.

Rather than classifying an entire image as real or fake, PatchGAN evaluates local image patches. The discriminator receives the luminance channel together with the corresponding color channels, encouraging the generator to produce locally realistic colors.

## Training

The generator is trained using a weighted combination of:

- L1 loss for pixel-level color accuracy
- Adversarial loss for realistic color generation
- SSIM loss for structural consistency

The generator objective is:

    L_G = L_GAN + 75L_1 + 50(1 - SSIM)

The discriminator is trained simultaneously to distinguish between real and generated colorizations.

## Dataset

The project uses the COCO 2017 training and validation datasets.

Images are:

- Resized to 256 × 256
- Randomly horizontally flipped during training
- Converted from RGB to LAB
- Normalized before training

For the current experiment, a randomly sampled 10% subset of the COCO training dataset is used.

### Download Dataset

    wget http://images.cocodataset.org/zips/train2017.zip
    wget http://images.cocodataset.org/zips/val2017.zip

    unzip train2017.zip
    unzip val2017.zip

## Training Configuration

| Parameter | Value |
|---|---:|
| Image Size | 256 × 256 |
| Batch Size | 32 |
| Epochs | 10 |
| Learning Rate | 0.0002 |
| Optimizer | Adam |
| Generator | ResNet-34 + U-Net |
| Discriminator | PatchGAN |
| Dataset | COCO 2017 |
| Training Subset | 10% |
| L1 Weight | 75 |
| SSIM Weight | 50 |

## Evaluation

The model is evaluated on the COCO validation set using L1 loss and SSIM.

### L1 Loss

Measures the difference between the predicted and ground-truth A/B color channels.

Lower values indicate better reconstruction accuracy.

### SSIM

Measures structural similarity between the predicted and ground-truth color channels.

Higher values indicate greater structural consistency.

The best generator checkpoint is selected based on validation SSIM.

## Results

The evaluation script generates visual comparisons between:

    Grayscale Input | Ground Truth | Predicted Colorization

The resulting image is saved as:

    colorization_results.png

These comparisons provide a direct way to inspect color accuracy, structural consistency, and artifacts produced by the model.

## TensorBoard

Training losses and generated images are logged using TensorBoard.

Start TensorBoard with:

    tensorboard --logdir=runs/colorization_gan

The training logs include:

- Generator loss
- Discriminator loss
- Generated images

## Outputs

The training process saves the best-performing models as well as checkpoints from each epoch.

    best_colorization_unet.pth
    best_patchgan_discriminator.pth

    colorization_unet_epoch_1.pth
    patchgan_discriminator_epoch_1.pth
    ...
    colorization_unet_epoch_10.pth
    patchgan_discriminator_epoch_10.pth

    colorization_results.png

## Setup

Install the required dependencies:

    pip install torch torchvision numpy matplotlib scikit-image pillow tensorboard

Update the dataset paths in the script if necessary:

    train_dir = "/content/train2017"
    val_dir = "/content/val2017"

Then run:

    python colorization_gan.py

The script trains the generator and discriminator, evaluates the model after each epoch, saves checkpoints, and generates example colorizations.

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- scikit-image
- Matplotlib
- TensorBoard
- ResNet-34
- U-Net
- PatchGAN
- Conditional GANs
- LAB color space
