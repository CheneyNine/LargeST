import math

import torch
import torch.nn as nn

from src.base.model import BaseModel


class RevIN(nn.Module):
    def __init__(self, num_features, affine=True, subtract_last=False, eps=1e-5):
        super().__init__()
        self.num_features = int(num_features)
        self.affine = bool(affine)
        self.subtract_last = bool(subtract_last)
        self.eps = float(eps)

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.num_features))

    def normalize(self, x):
        # x: [B, T, C]
        if self.subtract_last:
            center = x[:, -1:, :]
        else:
            center = x.mean(dim=1, keepdim=True)
        std = x.var(dim=1, keepdim=True, unbiased=False).add(self.eps).sqrt()
        x = (x - center) / std
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x, center, std

    def denormalize(self, x, center, std):
        # x: [B, H, C]
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        return x * std + center


class PatchTST(BaseModel):
    """
    PatchTST-style temporal patch transformer adapted to LargeST tensors.

    Reference:
    - PatchTST: A Time Series is Worth 64 Words
    - Official repo: https://github.com/yuqinie98/PatchTST
    """

    def __init__(
        self,
        traffic_dim,
        d_model,
        n_heads,
        e_layers,
        d_ff,
        patch_len,
        stride,
        dropout,
        fc_dropout,
        head_dropout,
        revin,
        affine,
        subtract_last,
        **args
    ):
        super(PatchTST, self).__init__(**args)

        if traffic_dim <= 0 or traffic_dim > self.input_dim:
            raise ValueError(
                "traffic_dim must be in (0, input_dim], got {} with input_dim={}".format(
                    traffic_dim, self.input_dim
                )
            )
        if patch_len <= 0 or stride <= 0:
            raise ValueError("patch_len and stride must be > 0")
        if self.seq_len + stride < patch_len:
            raise ValueError(
                "seq_len + stride must be >= patch_len, got {} + {} < {}".format(
                    self.seq_len, stride, patch_len
                )
            )

        self.traffic_dim = int(traffic_dim)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.e_layers = int(e_layers)
        self.d_ff = int(d_ff)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.revin_enabled = bool(revin)

        self.patch_num = int(
            math.floor((self.seq_len + self.stride - self.patch_len) / self.stride) + 1
        )
        if self.patch_num <= 0:
            raise ValueError("Invalid patch settings produce non-positive patch_num")

        self.revin = RevIN(
            num_features=self.traffic_dim,
            affine=bool(affine),
            subtract_last=bool(subtract_last),
        )
        self.patch_proj = nn.Linear(self.patch_len * self.traffic_dim, self.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_num, self.d_model) * 0.02)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.e_layers)
        self.encoder_norm = nn.LayerNorm(self.d_model)

        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(fc_dropout),
            nn.Linear(self.patch_num * self.d_model, self.horizon * self.output_dim),
            nn.Dropout(head_dropout),
        )

    def _patchify(self, series):
        # series: [S, T, C]
        if self.stride > 0:
            pad = series[:, -1:, :].expand(-1, self.stride, -1)
            series = torch.cat([series, pad], dim=1)

        patches = series.transpose(1, 2).unfold(dimension=2, size=self.patch_len, step=self.stride)
        # [S, C, P, patch_len] -> [S, P, C * patch_len]
        patches = patches.permute(0, 2, 1, 3).contiguous()
        return patches.view(series.shape[0], self.patch_num, self.traffic_dim * self.patch_len)

    def forward(self, inputs, label=None):
        del label
        traffic_inputs = inputs[..., : self.traffic_dim]
        batch_size, _, node_num, _ = traffic_inputs.shape
        if node_num != self.node_num:
            raise ValueError(
                "Node num mismatch in PatchTST forward: got {}, expected {}".format(
                    node_num, self.node_num
                )
            )

        series = traffic_inputs.permute(0, 2, 1, 3).contiguous()
        series = series.view(batch_size * node_num, self.seq_len, self.traffic_dim)

        if self.revin_enabled:
            series, center, scale = self.revin.normalize(series)
        else:
            center = scale = None

        patches = self._patchify(series)
        tokens = self.patch_proj(patches)
        tokens = self.input_dropout(tokens + self.pos_embedding)
        tokens = self.encoder(tokens)
        tokens = self.encoder_norm(tokens)

        prediction = self.head(tokens)
        prediction = prediction.view(batch_size * node_num, self.horizon, self.output_dim)

        if self.revin_enabled:
            prediction = self.revin.denormalize(
                prediction,
                center[..., : self.output_dim],
                scale[..., : self.output_dim],
            )

        prediction = prediction.view(batch_size, node_num, self.horizon, self.output_dim)
        prediction = prediction.permute(0, 2, 1, 3).contiguous()
        return prediction
