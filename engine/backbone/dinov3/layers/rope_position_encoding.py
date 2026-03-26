# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn


# RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
# Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}
        
        # 关键修改：确保 H 和 W 是 Python int 而不是 tensor
        # 如果 H 或 W 是 tensor，先转换为 Python int
        if torch.is_tensor(H):
            H = H.item()
        if torch.is_tensor(W):
            W = W.item()
        
        # 简化归一化方式，避免条件判断
        # 直接使用 separate 归一化，这是最稳定的
        coords_h = torch.arange(0.5, H, **dd) / H
        coords_w = torch.arange(0.5, W, **dd) / W
        
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0
        
        # 完全移除训练时的随机操作
        # 在 ONNX 导出时不需要这些
        
        # 确保 periods 是常数
        periods = self.periods
        
        # 计算角度 - 使用更稳定的方式
        # 避免使用 None 索引，改为显式 reshape
        D_head_half = self.D_head // 2
        D_head_quarter = self.D_head // 4
        
        # 重塑 coords 以便广播
        coords_reshaped = coords[:, :, None]  # [HW, 2, 1]
        periods_reshaped = periods[None, None, :]  # [1, 1, D//4]
        
        angles = 2 * math.pi * coords_reshaped / periods_reshaped  # [HW, 2, D//4]
        angles = angles.reshape(-1, D_head_half)  # [HW, D//2]
        angles = torch.cat([angles, angles], dim=-1)  # [HW, D]
        
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        return (sin, cos)

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
            )  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
        self.periods.data = periods
