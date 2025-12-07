import torch
import torchvision
import os
from torchvision import transforms
from PIL import Image
from pytorch_fid import fid_score
import argparse
import json

def save_real_images(args):
    """
    Load MNIST training set and save images to disk for FID calculation.
    Only saves if the directory doesn't exist or is empty.
    """
    dataset_name = 'mnist'
    if "cifar10" in args.config_path:
        dataset_name = 'cifar10'

    real_samples_dir = f'/scratch/scholar/{os.getenv("USER")}/diffusion/{dataset_name}/real_samples'
    
    # Check if real samples already exist
    if os.path.exists(real_samples_dir) and len(os.listdir(real_samples_dir)) > 0:
        print(f"Real samples already exist in {real_samples_dir}. Skipping extraction.")
        return real_samples_dir
    
    print(f"Saving {dataset_name} training set images to {real_samples_dir}...")
    os.makedirs(real_samples_dir, exist_ok=True)
    
    # Load MNIST training set
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    mnist_test = torchvision.datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transform
    )

    if mnist_test == 'cifar10':
       mnist_test = torchvision.datasets.CIFAR10(root="cifar10_data", train=True, download=True, transform=transform)
    
    # Save all test images
    for idx in range(len(mnist_test)):
        img_tensor, _ = mnist_test[idx]
        
        # Convert tensor to PIL Image (grayscale)
        img_tensor = img_tensor.squeeze(0)  # Remove channel dimension
        img = transforms.ToPILImage()(img_tensor)
        
        # Save as PNG
        img.save(os.path.join(real_samples_dir, f'real_{idx:05d}.png'))
        
        if (idx + 1) % 1000 == 0:
            print(f"Saved {idx + 1}/{len(mnist_test)} images")
    
    print(f"Saved {len(mnist_test)} real images to {real_samples_dir}")
    return real_samples_dir


def calculate_fid(args):
    """
    Calculate FID score between generated samples and real MNIST training images.
    """
    generated_samples_dir = f'/scratch/scholar/{os.getenv("USER")}/diffusion/{args.config_path}/generated_samples'
    
    # Ensure generated samples exist
    if not os.path.exists(generated_samples_dir):
        raise FileNotFoundError(f"Generated samples directory not found: {generated_samples_dir} \n Add generated samples to the directory.")
    
    num_generated = len([f for f in os.listdir(generated_samples_dir) if f.endswith('.png')])
    if num_generated == 0:
        raise ValueError(f"No images found in {generated_samples_dir}")
    
    print(f"Found {num_generated} generated samples in {generated_samples_dir}")
    
    # Save real images if needed
    real_samples_dir = save_real_images(args)
    
    num_real = len([f for f in os.listdir(real_samples_dir) if f.endswith('.png')])
    print(f"Using {num_real} real images from {real_samples_dir}")
    
    # Calculate FID score
    print("\nCalculating FID score...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    fid_value = fid_score.calculate_fid_given_paths(
        [generated_samples_dir, real_samples_dir],
        batch_size=50,
        device=device,
        dims=2048
    )
    
    print(f"\n{'='*50}")
    print(f"FID Score: {fid_value:.2f}")
    print(f"{'='*50}")
    
    return fid_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='mnist', type=str)
    args = parser.parse_args()
    calculate_fid(args)

