import torch
import torch.nn as nn
import math
import numpy as np


def get_timestep_embedding(timesteps, embedding_dim):

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device,) * -emb)
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

        self.attention_blocks = []
        for i in range(num_layers):
            attention_block = AttentionBlock(channels=out_channels, num_heads=num_heads)
            self.attention_blocks.append(attention_block)
        self.attention_blocks = nn.ModuleList(self.attention_blocks)

        self.down_sample_layer = nn.Identity()
        if self.down_sample:
            self.down_sample_layer = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, t_embeddings):
        # x --> B x C_in x H x W
        out = x
        for i in range(self.num_layers):
            out = self.resnet_blocks[i](out, t_embeddings)
            out = self.attention_blocks[i](out)
        
        # skip --> B x C_out x H x W
        skip_connection = out
        out = self.down_sample_layer(out)
        return out, skip_connection

class MidBlock(nn.Module):
    def __init__(self, channels, t_emb_dim, num_heads=4, num_layers=1):
        super().__init__()

        self.resnet_1 = ResnetBlock(channels, channels, t_emb_dim)
        self.attention = AttentionBlock(channels, num_heads)
        self.resnet_2 = ResnetBlock(channels, channels, t_emb_dim)

    def forward(self, x, t_embeddings):
        out = x
        out = self.resnet_1(out, t_embeddings)
        out = self.attention(out)
        out = self.resnet_2(out, t_embeddings)

        return out
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample=True, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample

        combined_channels = 2 * in_channels
        self.resnet_blocks = []
        for i in range(num_layers):
            resnet_block = ResnetBlock(in_channels=combined_channels if i == 0 else out_channels, out_channels=out_channels, t_embedding_dim=t_emb_dim)
            self.resnet_blocks.append(resnet_block)
        self.resnet_blocks = nn.ModuleList(self.resnet_blocks)

        self.attention_blocks = []
        for i in range(num_layers):
            attention_block = AttentionBlock(channels=out_channels, num_heads=num_heads)
            self.attention_blocks.append(attention_block)
        self.attention_blocks = nn.ModuleList(self.attention_blocks)

        self.up_sample_layer = nn.Identity()
        if self.up_sample:
            self.up_sample_layer = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, skip_connection, t_embeddings):
        x = self.up_sample_layer(x)
        x = torch.cat([x, skip_connection], dim=1)

        out = x
        for i in range(self.num_layers):
            out = self.resnet_blocks[i](out, t_embeddings)
            out = self.attention_blocks[i](out)
        
        return out


class Unet(nn.Module):
    def __init__(self, im_channels, down_channels, mid_channels, t_emb_dim, down_sample,
                downblock_layers, midblock_layers, upblock_layers, num_heads=4):
        super().__init__()

        self.im_channels = im_channels
        self.down_channels = down_channels
        self.mid_channels = mid_channels
        self.up_channels = list(reversed(down_channels))
        self.t_emb_dim = t_emb_dim
        self.down_sample = down_sample
        self.up_sample = list(reversed(down_sample))
        self.downblock_layers = downblock_layers
        self.midblock_layers = midblock_layers
        self.upblock_layers = upblock_layers
        self.num_heads = num_heads


        self.t_proj_init = nn.Sequential(
                            nn.Linear(self.t_emb_dim, self.t_emb_dim),
                            nn.SiLU(),
                            nn.Linear(self.t_emb_dim, self.t_emb_dim)
                        )

        self.conv_in = nn.Conv2d(self.im_channels, self.down_channels[0], kernel_size=3, stride=1, padding=(1, 1))

        self.down_blocks = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            down_block = DownBlock(self.down_channels[i], self.down_channels[i+1], self.t_emb_dim,
                                   self.down_sample[i], self.num_heads, self.downblock_layers)
            self.down_blocks.append(down_block)
        
        self.mid_blocks = nn.ModuleList([])
        for i in range(len(self.mid_channels)):
            mid_block = MidBlock(self.mid_channels[i], self.t_emb_dim, self.num_heads, self.midblock_layers)
            self.mid_blocks.append(mid_block)
        
        self.up_blocks = nn.ModuleList([])
        for i in range(len(self.up_channels)-1):
            up_block = UpBlock(self.up_channels[i], self.up_channels[i+1], self.t_emb_dim,
                               self.up_sample[i], self.num_heads, self.upblock_layers)
            self.up_blocks.append(up_block)
        
        self.norm_out = nn.GroupNorm(8, self.up_channels[-1])
        self.activation_out = nn.SiLU()
        self.conv_out = nn.Conv2d(self.up_channels[-1], self.im_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x, t):
        out = x
        
        out = self.conv_in(out)

        t_embeddings = get_timestep_embedding(torch.as_tensor(t), self.t_emb_dim)
        t_embeddings = self.t_proj_init(t_embeddings)

        skip_connections = []

        print(len(self.down_blocks))
        for down_block in self.down_blocks:
            out, skip_connection = down_block(out, t_embeddings)
            skip_connections.append(skip_connection.detach().clone())
        
        for mid_block in self.mid_blocks:
            out = mid_block(out, t_embeddings)
        
        for up_block in self.up_blocks:
            skip_connection = skip_connections.pop()
            out = up_block(out, t_embeddings, skip_connection)

        out = self.norm_out(out)
        out = self.activation_out(out)
        out = self.conv_out(out)

        return out
