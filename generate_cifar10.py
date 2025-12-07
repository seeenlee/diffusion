import torch
import torchvision
import argparse
import os
import json
from torchvision.utils import make_grid
from tqdm import tqdm
import numpy as np
from models.model import Unet
from scheduler.scheduler import Scheduler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

is_batch_job = 'SLURM_JOB_ID' in os.environ


def generate_with_progress(model, scheduler, train_config, model_config, diffusion_config):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions every 10%
    """
    xt = torch.randn((train_config['num_samples_progress'],
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size'])).to(device)
    
    save_steps = np.linspace(start=0, stop=diffusion_config['num_timesteps']-1, num=11, dtype=int)
    for i in tqdm(reversed(range(diffusion_config['num_timesteps'])), disable=is_batch_job):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, torch.as_tensor(i).to(device), noise_pred)
        
        # save predicted x_t every 10% of num_timesteps
        if i in save_steps:
            ims = torch.clamp(xt, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2
            grid = make_grid(ims, nrow=train_config['num_grid_rows'])
            img = torchvision.transforms.ToPILImage()(grid)
            if not os.path.exists(os.path.join(train_config['task_name'], 'progress_samples')):
                os.mkdir(os.path.join(train_config['task_name'], 'progress_samples'))
            img.save(os.path.join(train_config['task_name'], 'progress_samples', f"x0_{i}.png"))
            img.close()

def generate(model, scheduler, train_config, model_config, diffusion_config):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the final generated images
    """
    final_sample_dir = os.path.join(train_config['task_name'], 'generated_samples')
    os.makedirs(final_sample_dir, exist_ok=True)
    
    existing_files = [f for f in os.listdir(final_sample_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    num_files_existing = len(existing_files)
    print(f"Found {num_files_existing} images in {final_sample_dir}")

    # Generate in batches to avoid OOM
    batch_size = train_config.get('generation_batch_size', 100)  # Default to 100 if not specified
    num_samples = train_config['num_samples']
    remaining_samples = num_samples - num_files_existing
    num_batches = (remaining_samples + batch_size - 1) // batch_size  # Ceiling division
    
    sample_idx = num_files_existing
    print(f"Starting from sample {sample_idx:04d}")
    for batch_num in tqdm(range(num_batches), desc="Generating batches", disable=is_batch_job):
        # Calculate current batch size (last batch might be smaller)
        current_batch_size = min(batch_size, num_samples - batch_num * batch_size)
        
        xt = torch.randn((current_batch_size,
                          model_config['im_channels'],
                          model_config['im_size'],
                          model_config['im_size'])).to(device)
        
        for i in reversed(range(diffusion_config['num_timesteps'])):
            # predicted noise
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
            
            # Use scheduler to get x0 and xt-1
            xt, x0_pred = scheduler.sample_prev_timestep(xt, torch.as_tensor(i).to(device), noise_pred)
        
        gen_images = torch.clamp(xt, -1., 1.).detach().cpu()
        gen_images = (gen_images + 1) / 2

        for i in range(gen_images.shape[0]):
            img = gen_images[i]
            img = torchvision.transforms.ToPILImage()(img)
            img.save(os.path.join(final_sample_dir, f'sample_{sample_idx:04d}.png'))
            img.close()
            sample_idx += 1
    


def setup(args):
    # Read the config file #
    with open(args.config_path, "r") as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON file")
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # Load model with checkpoint
    model = Unet(model_config['im_channels'], 
                 model_config['down_channels'], 
                 model_config['mid_channels'], 
                 model_config['time_emb_dim'], 
                 model_config['down_sample'], 
                 model_config['num_down_layers'], 
                 model_config['num_mid_layers'], 
                 model_config['num_up_layers'], 
                 model_config['num_heads']).to(device)
    checkpoint = torch.load(os.path.join(train_config['task_name'],
                                         train_config['ckpt_name']), 
                           map_location=device, 
                           weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create the noise scheduler
    scheduler = Scheduler(diffusion_config['num_timesteps'],
                          diffusion_config['beta_start'],
                          diffusion_config['beta_end'])
    with torch.no_grad():
        if args.generate_progress:
            generate_with_progress(model, scheduler, train_config, model_config, diffusion_config)
        if args.generate_samples:
            generate(model, scheduler, train_config, model_config, diffusion_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/cifar10.json', type=str)
    parser.add_argument('--generate_progress', action='store_true', help='generate and save images along with their progress over timesteps')
    parser.add_argument('--generate_samples', action='store_true', help='generate and save final images')
    args = parser.parse_args()
    setup(args)