"""
Generator model for Pix2Pix using U-Net architecture.
"""

import torch
import torch.nn as nn
from src.components.config import DEVICE, logger
from src.logger import model_logger
from src.exceptions import ModelLoadError

class Block(nn.Module):
    """
    Convolutional block for U-Net encoder/decoder.
    """

    def __init__(self, in_channels: int, out_channels: int, down: bool = True,
                 act: str = "relu", use_dropout: bool = False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                     bias=False, padding_mode="reflect") if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    """
    U-Net generator for Pix2Pix.
    """

    def __init__(self, in_channels: int = 3, features: int = 64):
        super().__init__()
        model_logger.info(f"Initializing Generator with {in_channels} input channels and {features} features")

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        # Encoder
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1),
            nn.ReLU()
        )

        # Decoder with skip connections
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False)
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False)
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Output in range [-1, 1]
        )

    def forward(self, x):
        # Encoder
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)

        # Bottleneck
        bottleneck = self.bottleneck(d7)

        # Decoder with skip connections
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))

        return self.final_up(torch.cat([up7, d1], 1))

    def save(self, filepath: str):
        """Save model state."""
        try:
            torch.save(self.state_dict(), filepath)
            model_logger.info(f"Generator saved to {filepath}")
        except Exception as e:
            model_logger.error(f"Failed to save generator: {e}")
            raise ModelLoadError(f"Failed to save model: {e}")

    def load(self, filepath: str):
        """Load model state."""
        try:
            self.load_state_dict(torch.load(filepath, map_location=DEVICE))
            model_logger.info(f"Generator loaded from {filepath}")
        except Exception as e:
            model_logger.error(f"Failed to load generator: {e}")
            raise ModelLoadError(f"Failed to load model: {e}")


def test():
    """Test the generator model."""
    try:
        x = torch.randn((1, 3, 256, 256)).to(DEVICE)
        model = Generator(in_channels=3, features=64).to(DEVICE)
        preds = model(x)
        model_logger.info(f"Generator test passed. Input shape: {x.shape}, Output shape: {preds.shape}")
        assert preds.shape == (1, 3, 256, 256), f"Expected (1, 3, 256, 256), got {preds.shape}"
    except Exception as e:
        model_logger.error(f"Generator test failed: {e}")
        raise
