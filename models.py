import torch
from torch import nn


class MnaseEtienneNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv1d(4, 64, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Conv1d(64, 16, 8, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            nn.Conv1d(16, 80, 8, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(80),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(80 * 250, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 2)
        return self.conv_stack(x)


class ResidualConcatLayer(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.layer(x)], dim=1)


class ResidualAddLayer(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layer(x)


class PooledConvLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int
    ) -> None:
        super().__init__()
        self.pooled_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pooled_conv(x)


class DilatedConvLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ) -> None:
        super().__init__()
        self.dilated_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding="same",
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dilated_conv(x)


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ) -> None:
        super().__init__()
        self.dilated_conv = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding="same",
                dilation=dilation,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dilated_conv(x)


class Crop1d(nn.Module):
    def __init__(self, left: int, right: int) -> None:
        super().__init__()
        self.slice = slice(left, -right)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.slice]


class BassenjiEtienneNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            self.pooled_conv_wrapper(4, 32, 12, pool_size=8),
            self.pooled_conv_wrapper(32, 32, 5, pool_size=4),
            self.pooled_conv_wrapper(32, 32, 5, pool_size=4),
            self.dilated_conv_wrapper(32, 16, 5, dilation=2),
            ResidualConcatLayer(self.dilated_conv_wrapper(16, 16, 5, dilation=4)),
            ResidualConcatLayer(self.dilated_conv_wrapper(32, 16, 5, dilation=8)),
            ResidualConcatLayer(self.dilated_conv_wrapper(48, 16, 5, dilation=16)),
            nn.Conv1d(64, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

    def pooled_conv_wrapper(
        self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.2),
        )

    def dilated_conv_wrapper(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding="same",
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 2)
        return self.conv_stack(x)


class BassenjiMnaseNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            self.pooled_conv_wrapper(4, 32, 12, pool_size=4),
            self.pooled_conv_wrapper(32, 32, 5, pool_size=2),
            self.pooled_conv_wrapper(32, 32, 5, pool_size=2),
            self.dilated_conv_wrapper(32, 16, 5, dilation=2),
            ResidualConcatLayer(self.dilated_conv_wrapper(16, 16, 5, dilation=4)),
            ResidualConcatLayer(self.dilated_conv_wrapper(32, 16, 5, dilation=8)),
            ResidualConcatLayer(self.dilated_conv_wrapper(48, 16, 5, dilation=16)),
            nn.Conv1d(64, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

    def pooled_conv_wrapper(
        self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.2),
        )

    def dilated_conv_wrapper(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding="same",
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 2)
        return self.conv_stack(x)


class BassenjiMultiNetwork(nn.Module):
    def __init__(self, n_tracks: int = 2) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            PooledConvLayer(4, 32, 12, pool_size=4),
            PooledConvLayer(32, 32, 5, pool_size=2),
            PooledConvLayer(32, 32, 5, pool_size=2),
            DilatedConvLayer(32, 16, 5, dilation=2),
            ResidualConcatLayer(DilatedConvLayer(16, 16, 5, dilation=4)),
            ResidualConcatLayer(DilatedConvLayer(32, 16, 5, dilation=8)),
            ResidualConcatLayer(DilatedConvLayer(48, 16, 5, dilation=16)),
            nn.Conv1d(64, n_tracks, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 2)
        x = self.conv_stack(x)
        return torch.transpose(x, 1, 2)


class BassenjiMultiNetwork2(nn.Module):
    def __init__(self, n_tracks: int = 2) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            PooledConvLayer(4, 64, 12, pool_size=4),
            PooledConvLayer(64, 64, 5, pool_size=2),
            PooledConvLayer(64, 64, 5, pool_size=2),
            DilatedConvLayer(64, 32, 5, dilation=2),
            ResidualConcatLayer(DilatedConvLayer(32, 32, 5, dilation=4)),
            ResidualConcatLayer(DilatedConvLayer(64, 32, 5, dilation=8)),
            ResidualConcatLayer(DilatedConvLayer(96, 32, 5, dilation=16)),
            nn.Conv1d(128, n_tracks, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 2)
        x = self.conv_stack(x)
        return torch.transpose(x, 1, 2)


class BassenjiMultiNetworkCrop(nn.Module):
    def __init__(self, n_tracks: int = 2) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            PooledConvLayer(4, 32, 12, pool_size=4),
            PooledConvLayer(32, 32, 5, pool_size=2),
            PooledConvLayer(32, 32, 5, pool_size=2),
            DilatedConvLayer(32, 16, 5, dilation=2),
            ResidualConcatLayer(DilatedConvLayer(16, 16, 5, dilation=4)),
            ResidualConcatLayer(DilatedConvLayer(32, 16, 5, dilation=8)),
            ResidualConcatLayer(DilatedConvLayer(48, 16, 5, dilation=16)),
            Crop1d(8, 8),
            nn.Conv1d(64, n_tracks, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 2)
        x = self.conv_stack(x)
        return torch.transpose(x, 1, 2)


class OriginalBassenjiMultiNetwork(nn.Module):
    def __init__(self, n_tracks: int = 2) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            ConvBlock(4, 24, 12, 1),
            nn.MaxPool1d(2),
            ConvBlock(24, 32, 5, 1),
            nn.MaxPool1d(2),
            ConvBlock(32, 32, 5, 1),
            nn.MaxPool1d(2),
            ConvBlock(32, 32, 5, 1),
            nn.MaxPool1d(2),
            ConvBlock(32, 32, 5, 1),
            ResidualAddLayer(
                nn.Sequential(
                    ConvBlock(32, 16, 3, 4), ConvBlock(16, 32, 1, 1), nn.Dropout(0.3)
                )
            ),
            ResidualAddLayer(
                nn.Sequential(
                    ConvBlock(32, 16, 3, 8), ConvBlock(16, 32, 1, 1), nn.Dropout(0.3)
                )
            ),
            ResidualAddLayer(
                nn.Sequential(
                    ConvBlock(32, 16, 3, 16), ConvBlock(16, 32, 1, 1), nn.Dropout(0.3)
                )
            ),
            Crop1d(8, 8),
            ConvBlock(32, 64, 1, 1),
            nn.Dropout(0.05),
            nn.Conv1d(64, n_tracks, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 2)
        x = self.conv_stack(x)
        return torch.transpose(x, 1, 2)


class OriginalBassenjiMultiNetwork2(nn.Module):
    def __init__(self, n_tracks: int = 2) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            ConvBlock(4, 24, 12, 1),
            nn.MaxPool1d(4),
            ConvBlock(24, 32, 5, 1),
            nn.MaxPool1d(2),
            ConvBlock(32, 32, 5, 1),
            nn.MaxPool1d(2),
            ConvBlock(32, 32, 5, 1),
            ResidualAddLayer(
                nn.Sequential(
                    ConvBlock(32, 16, 3, 4), ConvBlock(16, 32, 1, 1), nn.Dropout(0.3)
                )
            ),
            ResidualAddLayer(
                nn.Sequential(
                    ConvBlock(32, 16, 3, 8), ConvBlock(16, 32, 1, 1), nn.Dropout(0.3)
                )
            ),
            ResidualAddLayer(
                nn.Sequential(
                    ConvBlock(32, 16, 3, 16), ConvBlock(16, 32, 1, 1), nn.Dropout(0.3)
                )
            ),
            Crop1d(8, 8),
            ConvBlock(32, 64, 1, 1),
            nn.Dropout(0.05),
            nn.Conv1d(64, n_tracks, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 2)
        x = self.conv_stack(x)
        return torch.transpose(x, 1, 2)


class OriginalBassenjiMultiNetworkNoCrop(nn.Module):
    def __init__(self, n_tracks: int = 2) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            ConvBlock(4, 24, 12, 1),
            nn.MaxPool1d(2),
            ConvBlock(24, 32, 5, 1),
            nn.MaxPool1d(2),
            ConvBlock(32, 32, 5, 1),
            nn.MaxPool1d(2),
            ConvBlock(32, 32, 5, 1),
            nn.MaxPool1d(2),
            ConvBlock(32, 32, 5, 1),
            ResidualAddLayer(
                nn.Sequential(
                    ConvBlock(32, 16, 3, 4), ConvBlock(16, 32, 1, 1), nn.Dropout(0.3)
                )
            ),
            ResidualAddLayer(
                nn.Sequential(
                    ConvBlock(32, 16, 3, 8), ConvBlock(16, 32, 1, 1), nn.Dropout(0.3)
                )
            ),
            ResidualAddLayer(
                nn.Sequential(
                    ConvBlock(32, 16, 3, 16), ConvBlock(16, 32, 1, 1), nn.Dropout(0.3)
                )
            ),
            # Crop1d(8, 8),
            ConvBlock(32, 64, 1, 1),
            nn.Dropout(0.05),
            nn.Conv1d(64, n_tracks, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 2)
        x = self.conv_stack(x)
        return torch.transpose(x, 1, 2)


class InceptionModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels_3: int,
                 out_channels_6: int,
                 out_channels_9: int,
                 pool_size: int = 2) -> None:
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_3, 3, padding="same"),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_6, 6, padding="same"),
            nn.ReLU(),
        )
        self.conv9 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_9, 9, padding="same"),
            nn.ReLU(),
        )
        self.pool_block = nn.Sequential(
            nn.MaxPool1d(pool_size),
            nn.BatchNorm1d(out_channels_3 + out_channels_6 + out_channels_9),
            nn.Dropout(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.concatenate([self.conv3(x), self.conv6(x), self.conv9(x)], dim=1)
        return self.pool_block(x)


class ConvNetwork(nn.Module):
    def __init__(self, length: int = 101) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            PooledConvLayer(4, 64, 6, pool_size=2),
            PooledConvLayer(64, 64, 6, pool_size=2),
        )
        self.dense_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(length // 4 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 2)
        x = self.conv_stack(x)
        x = torch.transpose(x, 1, 2)
        x = self.dense_stack(x)
        return x


class SiameseConvNetwork(nn.Module):
    def __init__(self, length: int = 101) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            PooledConvLayer(4, 64, 6, pool_size=2),
            PooledConvLayer(64, 64, 6, pool_size=2),
        )
        self.dense_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(length // 4 * 64 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        x1 = torch.transpose(input1, 1, 2)
        x1 = self.conv_stack(x1)
        x2 = torch.transpose(input2, 1, 2)
        x2 = self.conv_stack(x2)
        x = torch.concatenate([x1, x2], dim=1)
        x = torch.transpose(x, 1, 2)
        x = self.dense_stack(x)
        return x
    

class InceptionNetwork(nn.Module):
    def __init__(self, length: int = 101) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            InceptionModule(4, out_channels_3=32, out_channels_6=64, out_channels_9=16, pool_size=2),
            InceptionModule(112, out_channels_3=32, out_channels_6=64, out_channels_9=16, pool_size=2),
        )
        self.dense_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(length // 4 * 112, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 2)
        x = self.conv_stack(x)
        x = torch.transpose(x, 1, 2)
        x = self.dense_stack(x)
        return x


class SiameseInceptionNetwork(nn.Module):
    def __init__(self, length: int = 101) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            InceptionModule(4, out_channels_3=32, out_channels_6=64, out_channels_9=16, pool_size=2),
            InceptionModule(112, out_channels_3=32, out_channels_6=64, out_channels_9=16, pool_size=2),
        )
        self.dense_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(length // 4 * 112 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        x1 = torch.transpose(input1, 1, 2)
        x1 = self.conv_stack(x1)
        x2 = torch.transpose(input2, 1, 2)
        x2 = self.conv_stack(x2)
        x = torch.concatenate([x1, x2], dim=1)
        x = torch.transpose(x, 1, 2)
        x = self.dense_stack(x)
        return x


ARCHITECTURES = {
    "BassenjiMnaseNetwork": BassenjiMnaseNetwork,
    "BassenjiEtienneNetwork": BassenjiEtienneNetwork,
    "BassenjiMultiNetwork": BassenjiMultiNetwork,
    "BassenjiMultiNetwork2": BassenjiMultiNetwork2,
    "OriginalBassenjiMultiNetwork": OriginalBassenjiMultiNetwork,
    "OriginalBassenjiMultiNetwork2": OriginalBassenjiMultiNetwork2,
    "OriginalBassenjiMultiNetworkNoCrop": OriginalBassenjiMultiNetworkNoCrop,
    "BassenjiMultiNetworkCrop": BassenjiMultiNetworkCrop,
    "SiameseInceptionNetwork": SiameseInceptionNetwork,
    "InceptionNetwork": InceptionNetwork,
    "SiameseConvNetwork": SiameseConvNetwork,
    "ConvNetwork": ConvNetwork,
}
