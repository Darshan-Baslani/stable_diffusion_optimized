import torch
from torch import nn 
from torch.nn import functional as F

from attention import SelfAttention, CrossAttention


class TimeEmbedding:
    def __init__(self, n_embd: int):
        super().__init__()

        self.layer1 = nn.Linear(n_embd, 4*n_embd)
        self.layer2 = nn.Linear(4*n_embd, 4*n_embd)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # (1, 320)

        # (1, 320) -> (1, 4*320)
        time = self.layer1(time)

        time = F.silu(time)

        time = self.layer2(time)

        return time


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # (batch, channels, height, width) -> (batch, channels, height*2, width*2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)

        return x


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.norm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # (batch, 320, height/8, width/8) -> (batch, 4, height/8, width/8)
        x = self.norm(x)
        x = F.silu(x)
        x = self.conv(x)

        return x


class UNET_residualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int = 1280):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        # feature: (batch, in_channels, height, width)
        # time: (1, 1280)

        residue = feature

        feature = self.norm(feature)
        feature = F.silu(feature)
        feature = self.conv(feature)

        time = F.silu(time)
        time = self.time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.norm2(merged)
        merged = F.silu(merged)
        merged = self.conv2(merged)

        # (batch, out_channels, height, width)
        return merged + self.residual(residue)






class SwitchSequential(nn.Module):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_attentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_residualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoers = nn.Module([
            # (batch, 4, height/8, width/8) -> (batch, 320, height/8, width/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNET_residualBlock(320, 320), UNET_attnetionBlock(8, 40)),
            # (batch, 320, height/8, width/8) -> (batch, 320, height/16, width/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            # (batch, 320, height/16, width/16) -> (batch, 640, height/16, width/16)
            SwitchSequential(UNET_residualBlock(320, 640), UNET_attnetionBlock(8, 80)),
            SwitchSequential(UNET_residualBlock(640, 640), UNET_attnetionBlock(8, 80)),
            # (batch, 640, height/16, width/16) -> (batch, 640, height/32, width/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            # (batch, 640, height/32, width/32) -> (batch, 1280, height/32, width/32)
            SwitchSequential(UNET_residualBlock(640, 1280), UNET_attnetionBlock(8, 160)),
            SwitchSequential(UNET_residualBlock(1280, 1280), UNET_attnetionBlock(8, 160)),
            # (batch, 1280, height/32, width/32) -> (batch, 1280, height/64, width/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_residualBlock(1280, 1280)),
            SwitchSequential(UNET_residualBlock(1280, 1280))
        ])

        self.bottleneck = SwitchSequential(
            # (batch, 1280, height/64, width/64)
            UNET_ResidualBlock(1280, 1280), 
            UNET_AttentionBlock(8, 160), 
            UNET_ResidualBlock(1280, 1280), 
        )

        self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32) 
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])


class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (batch, 4, height/8, width/8)
        # context: (batch_dim, seq_len, dim(768)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (batch, 4, height/8, width/8) -> (batch, 320, height/8, width/8)
        output = self.unet(latent, context, time)

        # (batch, 320, height/8, width/8) -> (batch, 4, height/8, width/8)
        output = self.final(output)

        return output