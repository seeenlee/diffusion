import torch

class Scheduler:
    def __init__(self, num_steps, beta_start, beta_end):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alpha_cumprod)

    def add_noise(self, x_t, t, noise):
        x_t_shape = x_t.shape
        batch_size = x_t_shape[0]

        sqrt_alpha_cumprod = self.sqrt_alpha_cumprod.to(x_t.device)[t].reshape(batch_size)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(x_t.device)[t].reshape(batch_size)
        
        for _ in range(len(x_t_shape) - 1):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
        for _ in range(len(noise.shape) - 1):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.unsqueeze(-1)

        return sqrt_alpha_cumprod.to(x_t.device) * x_t + sqrt_one_minus_alphas_cumprod.to(x_t.device) * noise
    
    def sample_prev_timestep(self, x_t, t, noise):
        # Convert t to scalar integer if it's a tensor
        if isinstance(t, torch.Tensor):
            t = t.item()
        
        x0 = (x_t - (self.sqrt_one_minus_alphas_cumprod.to(x_t.device)[t] * noise)) / torch.sqrt(self.alpha_cumprod.to(x_t.device)[t])
        x0 = torch.clamp(x0, -1., 1.)

        mean = x_t - ((self.betas.to(x_t.device)[t]) * noise) / (self.sqrt_one_minus_alphas_cumprod.to(x_t.device)[t])
        mean = mean / torch.sqrt(self.alphas.to(x_t.device)[t])

        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_cumprod.to(x_t.device)[t-1]) / (1. - self.alpha_cumprod.to(x_t.device)[t])
            variance = variance * self.betas.to(x_t.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(x_t.shape).to(x_t.device)
            return mean + sigma * z, x0
