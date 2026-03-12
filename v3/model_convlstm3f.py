from typing import List, Tuple

import torch
import torch.nn as nn

try:
    from torchvision.models import resnet18
except Exception as exc:  # pragma: no cover
    resnet18 = None
    _TORCHVISION_IMPORT_ERROR = exc
else:
    _TORCHVISION_IMPORT_ERROR = None


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)

        i, f, g, o = torch.chunk(gates, chunks=4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: List[int], kernel_size: int = 3):
        super().__init__()
        if not hidden_channels:
            raise ValueError('hidden_channels must be non-empty')

        self.hidden_channels = list(hidden_channels)
        cells = []
        in_ch = input_channels
        for h_ch in self.hidden_channels:
            cells.append(ConvLSTMCell(in_ch, h_ch, kernel_size=kernel_size))
            in_ch = h_ch
        self.cells = nn.ModuleList(cells)

    def forward(self, x: torch.Tensor):
        # x: (B, T, C, H, W)
        if x.ndim != 5:
            raise ValueError(f'Expected input shape (B,T,C,H,W), got {x.shape}')

        b, t, _, h, w = x.shape
        device = x.device

        current = x
        last_states = []

        for layer_idx, cell in enumerate(self.cells):
            h_state = torch.zeros(b, cell.hidden_channels, h, w, device=device, dtype=x.dtype)
            c_state = torch.zeros(b, cell.hidden_channels, h, w, device=device, dtype=x.dtype)

            outputs = []
            for step in range(t):
                h_state, c_state = cell(current[:, step], (h_state, c_state))
                outputs.append(h_state)

            current = torch.stack(outputs, dim=1)
            last_states.append((h_state, c_state))

            h = current.shape[-2]
            w = current.shape[-1]

        return current, last_states


class ConvLSTM3FPolicy(nn.Module):
    def __init__(
        self,
        hidden_channels: List[int] = None,
        convlstm_kernel_size: int = 3,
        mlp_hidden_dim: int = 128,
    ):
        super().__init__()

        if resnet18 is None:
            raise RuntimeError(
                'torchvision is required for ConvLSTM3FPolicy. '
                f'Import error: {_TORCHVISION_IMPORT_ERROR}'
            )

        if hidden_channels is None:
            hidden_channels = [128, 128]

        backbone = resnet18(weights=None)
        self.shared_cnn = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        )

        cnn_out_channels = 128
        self.temporal = ConvLSTM(
            input_channels=cnn_out_channels,
            hidden_channels=hidden_channels,
            kernel_size=convlstm_kernel_size,
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(hidden_channels[-1], mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, 2),
            nn.Tanh(),
        )

        self.hidden_channels = list(hidden_channels)
        self.convlstm_kernel_size = int(convlstm_kernel_size)
        self.mlp_hidden_dim = int(mlp_hidden_dim)
        self.use_tanh = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1, H, W)
        if x.ndim != 5:
            raise ValueError(f'Expected input shape (B,T,1,H,W), got {x.shape}')

        b, t, c, h, w = x.shape
        if t != 3:
            raise ValueError(f'V3 model expects T=3 frames, got T={t}')
        if c != 1:
            raise ValueError(f'V3 model expects C=1 masks, got C={c}')

        frames = x.float().repeat(1, 1, 3, 1, 1)  # grayscale -> pseudo RGB
        frames = frames.view(b * t, 3, h, w)

        feat = self.shared_cnn(frames)
        _, c_feat, h_feat, w_feat = feat.shape
        feat_seq = feat.view(b, t, c_feat, h_feat, w_feat)

        _, last_states = self.temporal(feat_seq)
        last_hidden = last_states[-1][0]

        pooled = self.gap(last_hidden).flatten(1)
        action = self.head(pooled)
        return action
