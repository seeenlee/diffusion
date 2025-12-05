import torch
import argparse
import json
from scheduler.scheduler import Scheduler
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    with open(args.config, "r") as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON file")
    print(config)

    diffusion_params = config["diffusion_params"]
    dataset_params = config["dataset_params"]
    model_params = config["model_params"]
    train_params = config["train_params"]

    scheduler = Scheduler(diffusion_params["num_timesteps"], diffusion_params["beta_start"], diffusion_params["beta_end"])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", default="config/default.json", type=str)
    args = parser.parse_args()
    train(args)