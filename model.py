"""Conditional variational autoencoder for 128 x 128 chest X-rays."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _normalization(num_channels: int) -> nn.GroupNorm:
    """GroupNorm is stable even when the last training batch is small."""
    return nn.GroupNorm(num_groups=8, num_channels=num_channels)


class EncoderBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            _normalization(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )


class DecoderBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            _normalization(out_channels),
            nn.ReLU(inplace=True),
        )


class ConditionalDecoderBlock(nn.Module):
    """Upsampling block modulated by the class embedding."""

    def __init__(self, in_channels: int, out_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.convolution = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.normalization = nn.GroupNorm(
            num_groups=8, num_channels=out_channels, affine=False
        )
        self.condition_to_scale_shift = nn.Linear(condition_dim, 2 * out_channels)
        self.activation = nn.ReLU(inplace=True)

    def reset_conditioning(self) -> None:
        # Begin as regular GroupNorm, then learn class-specific modulation.
        nn.init.zeros_(self.condition_to_scale_shift.weight)
        nn.init.zeros_(self.condition_to_scale_shift.bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        features = self.normalization(self.convolution(features))
        scale, shift = self.condition_to_scale_shift(condition).chunk(2, dim=1)
        features = features * (1.0 + scale[:, :, None, None])
        features = features + shift[:, :, None, None]
        return self.activation(features)


class ConditionalVAE(nn.Module):
    """A convolutional VAE conditioned on the X-ray class label.

    The condition is supplied to both the approximate posterior and decoder.
    This makes q(z | x, y) and p(x | z, y) explicitly class-conditional.
    """

    def __init__(
        self,
        image_size: int = 128,
        latent_dim: int = 128,
        num_classes: int = 2,
        condition_dim: int = 32,
        architecture: str = "spatial_film",
    ) -> None:
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError("image_size must be divisible by 16")
        if architecture not in {"bottleneck", "spatial_film"}:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.condition_dim = condition_dim
        self.feature_size = image_size // 16
        self.architecture = architecture

        self.condition_embedding = nn.Embedding(num_classes, condition_dim)
        encoder_input_channels = 1 if architecture == "bottleneck" else 1 + num_classes
        self.encoder = nn.Sequential(
            EncoderBlock(encoder_input_channels, 32),
            EncoderBlock(32, 64),
            EncoderBlock(64, 128),
            EncoderBlock(128, 256),
        )

        encoded_dim = 256 * self.feature_size * self.feature_size
        self.to_mu = nn.Linear(encoded_dim + condition_dim, latent_dim)
        self.to_logvar = nn.Linear(encoded_dim + condition_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim + condition_dim, encoded_dim)
        if architecture == "bottleneck":
            # This branch preserves compatibility with checkpoints from the
            # initial experiment.
            self.decoder = nn.Sequential(
                DecoderBlock(256, 128),
                DecoderBlock(128, 64),
                DecoderBlock(64, 32),
                nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid(),
            )
        else:
            self.decoder_blocks = nn.ModuleList(
                [
                    ConditionalDecoderBlock(256, 128, condition_dim),
                    ConditionalDecoderBlock(128, 64, condition_dim),
                    ConditionalDecoderBlock(64, 32, condition_dim),
                ]
            )
            self.decoder_output = nn.Sequential(
                nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid(),
            )

        self.apply(self._initialize_weights)
        if architecture == "spatial_film":
            for block in self.decoder_blocks:
                block.reset_conditioning()

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.architecture == "spatial_film":
            label_maps = F.one_hot(labels, num_classes=self.num_classes).to(
                images.dtype
            )
            label_maps = label_maps[:, :, None, None].expand(
                -1, -1, images.shape[2], images.shape[3]
            )
            images = torch.cat((images, label_maps), dim=1)
        features = self.encoder(images).flatten(start_dim=1)
        condition = self.condition_embedding(labels)
        features = torch.cat((features, condition), dim=1)
        return self.to_mu(features), self.to_logvar(features)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return mu + noise * std

    def decode(self, latent: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        condition = self.condition_embedding(labels)
        hidden = self.decoder_input(torch.cat((latent, condition), dim=1))
        hidden = hidden.view(-1, 256, self.feature_size, self.feature_size)
        if self.architecture == "bottleneck":
            return self.decoder(hidden)
        for block in self.decoder_blocks:
            hidden = block(hidden, condition)
        return self.decoder_output(hidden)

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(images, labels)
        latent = self.reparameterize(mu, logvar)
        reconstruction = self.decode(latent, labels)
        return reconstruction, mu, logvar

    @torch.no_grad()
    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        latent = torch.randn(labels.shape[0], self.latent_dim, device=labels.device)
        return self.decode(latent, labels)
