"""Policy + value network — a small AlphaZero-style residual ConvNet.

A convolutional residual tower feeds two heads:
  - policy: a distribution over the N*N+1 actions (cells + swap), in canonical orientation,
  - value:  a scalar in [-1, 1], the side-to-move's expected game result.

Width/depth are config knobs (NetConfig) so the same code scales from a laptop smoke test to a VPS run.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SEBlock(nn.Module):
    """Squeeze-and-excitation: global average pool -> 2-layer MLP -> per-channel gates in [0,1].

    Injects board-wide context into every residual block. Valuable for Hex specifically, where a win is
    a global edge-to-edge connection that a stack of local 3x3 convs only perceives slowly."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))                  # (B, C) global average pool
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s[:, :, None, None]


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, use_se: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = _SEBlock(channels) if use_se else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.se is not None:
            x = self.se(x)
        return F.relu(x + residual)


class HexNet(nn.Module):
    def __init__(self, board_size: int, in_planes: int, channels: int, blocks: int, value_hidden: int,
                 use_se: bool = False) -> None:
        super().__init__()
        self.board_size = board_size
        self.num_actions = board_size * board_size + 1

        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[_ResidualBlock(channels, use_se) for _ in range(blocks)])

        # Policy head: 2 feature maps -> flat -> action logits (cells + swap).
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.num_actions)

        # Value head: 1 feature map -> hidden -> tanh scalar.
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.tower(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return policy_logits, value


def build_net(cfg) -> HexNet:
    return HexNet(
        board_size=cfg.game.board_size,
        in_planes=cfg.net.in_planes,
        channels=cfg.net.channels,
        blocks=cfg.net.blocks,
        value_hidden=cfg.net.value_hidden,
        use_se=cfg.net.use_se,
    )


_COMPILE_PREFIX = "_orig_mod."


def BareModule(net):
    """The underlying module, unwrapping torch.compile's OptimizedModule wrapper.

    torch.compile returns a wrapper holding the real module in ._orig_mod and its
    state_dict() keys gain an '_orig_mod.' prefix. Saving/serialising from the bare
    module keeps weights loadable by a plain (uncompiled) net.
    """
    return getattr(net, "_orig_mod", net)


def CleanStateDict(state: dict) -> dict:
    """Strip torch.compile's '_orig_mod.' key prefix so weights load into a plain net."""
    return {
        (k[len(_COMPILE_PREFIX):] if k.startswith(_COMPILE_PREFIX) else k): v
        for k, v in state.items()
    }
