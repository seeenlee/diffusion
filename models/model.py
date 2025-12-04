import torch
import torch.nn as nn
import math
import numpy as np


def get_timestep_embedding(timesteps, embedding_dim):

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, type=torch.float32, device=timesteps.device,) * -emb)
    emb = timesteps.type(torch.float32)[:, None] * emb[None, :]
    emb = torch.concat([torch.sin(emb), torch.cos(emb)], axis=1)

    if embedding_dim % 2 == 1:
        emb = torch.pad(emb, [0,1,0,0])

    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_embedding_dim):
        super().__init__()
        self.conv_1 = nn.Sequential(
                        nn.GroupNorm(8, in_channels),
                        nn.SiLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
                    )

        self.t_emb_layer = nn.Sequential(
                            nn.SiLU(),
                            nn.Linear(t_embedding_dim, out_channels)
                        )
        
        self.conv_2 = nn.Sequential(
                        nn.GroupNorm(8, out_channels),
                        nn.SiLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                    )

        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if in_channels == out_channels:
            self.residual_conv = nn.Identity()
    
    def forward(self, x, t_embedding):
        residual = x

        out = x

        #first convolution layer
        out = self.conv_1(out)

        # adding embeddings
        out += self.t_emb_layer(t_embedding)[:, :, None, None]

        # second convolution layer
        out = self.conv_2(out)

        # residual connection
        out += self.residual_conv(residual)

        return out

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape

        # reshaping and normalizing
        out = x.reshape(B, C, H*W)
        out = self.norm(out)
        out = out.transpose(1, 2)

        # self-attention calculation
        out, _ = self.attention(out, out, out)

        # reshaping attention output back to BxCxHxW
        out = out.transpose(1, 2)
        out = out.reshape(B, C, H, W)

        return x + out

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample=True, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample

        self.resnet_blocks = []
        for i in range(num_layers):
            resnet_block = ResnetBlock(in_channels=in_channels if i == 0 else out_channels, out_channels=out_channels, t_embedding_dim=t_emb_dim)
            self.resnet_blocks.append(resnet_block)
        self.resnet_blocks = nn.ModuleList(self.resnet_blocks)

        self.attention_block = AttentionBlock(channels=out_channels, num_heads=num_heads)

        self.down_sample_layer = nn.Identity()
        if self.down_sample:
            self.down_sample_layer = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, t_embeddings):
        out = x
        for i in range(self.num_layers):
            out = self.resnet_blocks[i](out, t_embeddings)
        
        out = self.attention_block(out)
        out = self.down_sample_layer(out)
        return out