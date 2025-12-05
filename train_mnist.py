import torch
import argparse
import json
from scheduler.scheduler import Scheduler
from torchvision import datasets, transforms
from models.model import Unet
from torch.optim import Adam
from tqdm import tqdm
from torch import nn
import numpy as np
import os
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def train(args):
    with open(args.config, "r") as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON file")
    print(config)

    diffusion_params = config["diffusion_params"]
    model_params = config["model_params"]
    train_params = config["train_params"]

    scheduler = Scheduler(diffusion_params["num_timesteps"], diffusion_params["beta_start"], diffusion_params["beta_end"])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=train_params["batch_size"], shuffle=True)

    model = Unet(model_params["im_channels"], model_params["down_channels"], model_params["mid_channels"], model_params["time_emb_dim"], model_params["down_sample"], model_params["num_down_layers"], model_params["num_mid_layers"], model_params["num_up_layers"], model_params["num_heads"]).to(device)

    model.train()

    num_epochs = train_params["num_epochs"]
    optimizer = Adam(model.parameters(), lr=train_params["lr"])
    criterion = nn.MSELoss()

    for epoch_idx in range(num_epochs):
        losses = []
        for im, _ in tqdm(train_loader):
            optimizer.zero_grad()
            im = im.float().to(device)

            noise = torch.randn_like(im).to(device)

            t = torch.randint(0, diffusion_params["num_timesteps"], (im.shape[0],)).to(device)

            noisy_im = scheduler.add_noise(im, t, noise)

            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"Epoch {epoch_idx+1}/{num_epochs}, Loss: {np.mean(losses)}")

        torch.save(model.state_dict(), os.path.join(train_params["task_name"],train_params["ckpt_name"]))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", default="config/default.json", type=str)
    args = parser.parse_args()
    train(args)