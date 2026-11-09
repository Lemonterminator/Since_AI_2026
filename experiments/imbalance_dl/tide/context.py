"""Daily context encoder for the split-horizon TiDE path.

The native TiDE history path will later see only the most recent 16 steps, so
distant daily structure must reach the encoding by another route.  This module
turns the full sanitized and masked day of target plus history features into
one context vector that the parent can add to the native encoding.

The encoder splits the day into chronological non-overlapping patches.  Each
patch of ``patch_size`` steps is flattened to ``patch_size * (1 + F)`` numeric
inputs, where channel 0 is the target and channels 1..F are the history
features.  One ``MLPResidual`` block encodes each patch independently, and a
linear projection maps the chronological patch encodings to ``width`` outputs.
"""

import torch
from torch import nn

from .native import MLPResidual


class DailyContext(nn.Module):
    """Encode a full daily history into one ``[B, width]`` context vector.

    Args:
        history: Number of history steps ``L``.  It must be positive and
            divisible by ``patch_size``.
        feature_channels: History-feature count ``F`` per step.
        width: Output context width.
        dropout: Dropout rate applied inside the patch encoder.
        patch_size: Number of steps per non-overlapping chronological patch.
    """

    def __init__(
        self,
        history: int,
        feature_channels: int,
        width: int,
        dropout: float = 0.0,
        patch_size: int = 4,
    ):
        super().__init__()
        history = int(history)
        feature_channels = int(feature_channels)
        patch_size = int(patch_size)
        if history <= 0:
            raise ValueError(f"history must be positive; got {history}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive; got {patch_size}")
        if history % patch_size != 0:
            raise ValueError(
                f"history {history} is not divisible by patch_size {patch_size}"
            )
        if feature_channels < 0:
            raise ValueError(
                f"feature_channels must be non-negative; got {feature_channels}"
            )

        self.history = history
        self.feature_channels = feature_channels
        self.patch_size = patch_size
        self.patch_count = history // patch_size
        self.patch_channels = patch_size * (1 + feature_channels)

        self.patch_encoder = MLPResidual(
            self.patch_channels,
            64,
            16,
            layer_norm=True,
            pre_norm=True,
            dropout_rate=dropout,
        )
        self.projection = nn.Linear(self.patch_count * 16, int(width))

    def forward(
        self,
        target: torch.Tensor,
        history_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return one context vector per batch item.

        ``target`` is ``[B, L]`` and ``history_features`` is ``[B, L, F]``.
        Both inputs are already sanitized and masked by the caller.
        """
        if target.dim() != 2:
            raise ValueError(f"target must be [B, L]; got {tuple(target.shape)}")
        batch_size, history = target.shape
        if history != self.history:
            raise ValueError(f"target has {history} steps; expected {self.history}")
        if history_features.shape != (
            batch_size,
            self.history,
            self.feature_channels,
        ):
            raise ValueError(
                f"history_features has shape {tuple(history_features.shape)}; "
                f"expected {(batch_size, self.history, self.feature_channels)}"
            )

        # Target first, then the history features: [B, L, 1 + F].
        sequence = torch.cat((target.unsqueeze(-1), history_features), dim=-1)
        patches = sequence.reshape(
            batch_size, self.patch_count, self.patch_channels
        )
        encoded = self.patch_encoder(patches)  # [B, patch_count, 16]
        flat = encoded.reshape(batch_size, self.patch_count * 16)
        return self.projection(flat)
