import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import json
from torchvision import datasets, transforms
import torchvision.io as io
from torch.utils.data import DataLoader
from ignite.metrics import FID, InceptionScore
import pytorch_fid_wrapper as pfw


def evaluate(args):
    image_directory = os.path.join(args.dataset, 'generated_samples')

    image_paths = [
        os.path.join(image_directory, f)
        for f in os.listdir(image_directory)
        if f.lower().endswith('.png')
    ]

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)) if args.dataset == 'mnist' else transforms.Identity(),
    ])

    img_list = []
    for path in image_paths:
        try:
            img = io.read_image(path)
            img = transforms(img)
            img_list.append(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

    img_list = torch.stack(img_list)

    train_dataset = None
    if args.dataset == 'mnist':
        train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transforms)
    elif args.dataset == 'mnist':
        train_dataset = datasets.CIFAR10(root="cifar10_data", train=True, download=True, transform=transforms)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    true_images = []
    for images, _ in train_loader:
        true_images.append(images)

    true_images = torch.cat(true_images, dim=0)

    config = pfw.Config(
        batch_size=64,
        dims=2048,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    fid_score = pfw.fid(true_images, img_list, config)

    print(f"FID Score for {args.dataset}: {fid_score}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for evaluation')
    parser.add_argument('--dataset', dest='dataset',
                        default='mnist', type=str)
    args = parser.parse_args()
    evaluate(args)