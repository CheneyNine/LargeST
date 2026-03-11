import math

import torch
import torch.nn as nn

from src.base.model import BaseModel


class _Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.attn(x, x, x, need_weights=False)
        return self.proj_dropout(out)


class _Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class WindowAttBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num, size, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.num = int(num)
        self.size = int(size)

        self.nnorm1 = nn.LayerNorm(hidden_size)
        self.nattn = _Attention(hidden_size, num_heads=num_heads, dropout=dropout)
        self.nnorm2 = nn.LayerNorm(hidden_size)
        self.nmlp = _Mlp(hidden_size, mlp_ratio=mlp_ratio, dropout=dropout)

        self.snorm1 = nn.LayerNorm(hidden_size)
        self.sattn = _Attention(hidden_size, num_heads=num_heads, dropout=dropout)
        self.snorm2 = nn.LayerNorm(hidden_size)
        self.smlp = _Mlp(hidden_size, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        batch_size, tpn, patch_nodes, hidden = x.shape
        patch_num = self.num
        patch_size = self.size
        if patch_num * patch_size != patch_nodes:
            raise ValueError(
                "WindowAttBlock shape mismatch: {} * {} != {}".format(
                    patch_num, patch_size, patch_nodes
                )
            )

        x = x.reshape(batch_size, tpn, patch_num, patch_size, hidden)

        depth_tokens = x.reshape(batch_size * tpn * patch_num, patch_size, hidden)
        depth_tokens = depth_tokens + self.sattn(self.snorm1(depth_tokens))
        depth_tokens = depth_tokens + self.smlp(self.snorm2(depth_tokens))
        x = depth_tokens.reshape(batch_size, tpn, patch_num, patch_size, hidden)

        breadth_tokens = x.transpose(2, 3).reshape(batch_size * tpn * patch_size, patch_num, hidden)
        breadth_tokens = breadth_tokens + self.nattn(self.nnorm1(breadth_tokens))
        breadth_tokens = breadth_tokens + self.nmlp(self.nnorm2(breadth_tokens))
        x = breadth_tokens.reshape(batch_size, tpn, patch_size, patch_num, hidden).transpose(2, 3)

        return x.reshape(batch_size, tpn, patch_nodes, hidden)


class PatchSTG(BaseModel):
    """
    LargeST adaptation of official PatchSTG.

    Core alignment:
    - temporal patch embedding on traffic + TOD + DOW
    - KDTree-based spatial patch reorder
    - depth and breadth attention over spatial patches
    - point-wise regression back to [B, H, N, 1]
    """

    def __init__(
        self,
        ori_parts_idx,
        reo_parts_idx,
        reo_all_idx,
        tem_patchsize,
        tem_patchnum,
        spa_patchsize,
        spa_patchnum,
        factors,
        layers,
        steps_per_day,
        day_of_week_size,
        input_emb_dims,
        node_dims,
        tod_dims,
        dow_dims,
        traffic_dim=1,
        time_day_idx=-1,
        day_in_week_idx=-1,
        dropout=0.1,
        **args
    ):
        super().__init__(**args)

        self.traffic_dim = int(traffic_dim)
        if self.traffic_dim != 1:
            raise ValueError(
                "PatchSTG currently supports flow-only forecasting in LargeST. "
                "Set traffic_dim=1, got {}".format(self.traffic_dim)
            )
        if self.traffic_dim > self.input_dim:
            raise ValueError(
                "traffic_dim must be <= input_dim, got {} > {}".format(
                    self.traffic_dim, self.input_dim
                )
            )

        self.tem_patchsize = int(tem_patchsize)
        self.tem_patchnum = int(tem_patchnum)
        self.spa_patchsize = int(spa_patchsize)
        self.spa_patchnum = int(spa_patchnum)
        self.factors = int(factors)
        self.layers = int(layers)
        self.steps_per_day = int(steps_per_day)
        self.day_of_week_size = int(day_of_week_size)
        self.time_day_idx = int(time_day_idx)
        self.day_in_week_idx = int(day_in_week_idx)

        if self.tem_patchsize * self.tem_patchnum != self.seq_len:
            raise ValueError(
                "PatchSTG requires tem_patchsize * tem_patchnum == seq_len, got {} * {} != {}".format(
                    self.tem_patchsize, self.tem_patchnum, self.seq_len
                )
            )
        if self.spa_patchnum % self.factors != 0:
            raise ValueError(
                "spa_patchnum must be divisible by factors, got {} % {} != 0".format(
                    self.spa_patchnum, self.factors
                )
            )

        self.register_buffer(
            "ori_parts_idx", torch.as_tensor(ori_parts_idx, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "reo_parts_idx", torch.as_tensor(reo_parts_idx, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "reo_all_idx", torch.as_tensor(reo_all_idx, dtype=torch.long), persistent=False
        )

        dims = int(input_emb_dims) + int(node_dims) + int(tod_dims) + int(dow_dims)
        self.hidden_dims = dims

        self.input_st_fc = nn.Conv2d(
            in_channels=3,
            out_channels=int(input_emb_dims),
            kernel_size=(1, self.tem_patchsize),
            stride=(1, self.tem_patchsize),
            bias=True,
        )
        self.node_emb = nn.Parameter(torch.empty(self.node_num, int(node_dims)))
        nn.init.xavier_uniform_(self.node_emb)

        self.time_in_day_emb = nn.Parameter(torch.empty(self.steps_per_day, int(tod_dims)))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, int(dow_dims)))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        patch_num_after_merge = self.spa_patchnum // self.factors
        patch_size_after_merge = self.spa_patchsize * self.factors
        self.spa_encoder = nn.ModuleList(
            [
                WindowAttBlock(
                    hidden_size=dims,
                    num_heads=1,
                    num=patch_num_after_merge,
                    size=patch_size_after_merge,
                    mlp_ratio=1.0,
                    dropout=dropout,
                )
                for _ in range(self.layers)
            ]
        )

        self.regression_conv = nn.Conv2d(
            in_channels=self.tem_patchnum * dims,
            out_channels=self.horizon,
            kernel_size=(1, 1),
            bias=True,
        )

    def _extract_tod(self, history_data):
        if self.time_day_idx < 0 or self.time_day_idx >= history_data.shape[-1]:
            raise ValueError("PatchSTG requires a valid time_day_idx")
        values = history_data[..., self.time_day_idx]
        if values.detach().max() <= 1.5 and values.detach().min() >= 0:
            tod_idx = torch.floor(values * self.steps_per_day).long()
            tod_norm = values
        else:
            tod_idx = values.long()
            tod_norm = tod_idx.float() / float(self.steps_per_day)
        tod_idx = torch.clamp(tod_idx, min=0, max=self.steps_per_day - 1)
        return tod_idx, tod_norm

    def _extract_dow(self, history_data):
        if self.day_in_week_idx < 0 or self.day_in_week_idx >= history_data.shape[-1]:
            raise ValueError("PatchSTG requires a valid day_in_week_idx")
        values = history_data[..., self.day_in_week_idx]
        if values.detach().max() <= 1.5 and values.detach().min() >= 0:
            dow_idx = torch.floor(values * self.day_of_week_size).long()
            dow_norm = values
        else:
            dow_idx = values.long()
            dow_norm = dow_idx.float() / float(self.day_of_week_size)
        dow_idx = torch.clamp(dow_idx, min=0, max=self.day_of_week_size - 1)
        return dow_idx, dow_norm

    def embedding(self, history_data):
        batch_size = history_data.shape[0]
        x = history_data[..., : self.traffic_dim]
        tod_idx, tod_norm = self._extract_tod(history_data)
        dow_idx, dow_norm = self._extract_dow(history_data)

        x1 = torch.cat([x, tod_norm.unsqueeze(-1), dow_norm.unsqueeze(-1)], dim=-1).float()
        input_data = self.input_st_fc(x1.transpose(1, 3)).transpose(1, 3)
        tpn = input_data.shape[1]

        tod_token_idx = tod_idx[:, -tpn:, :]
        dow_token_idx = dow_idx[:, -tpn:, :]
        input_data = torch.cat([input_data, self.time_in_day_emb[tod_token_idx]], dim=-1)
        input_data = torch.cat([input_data, self.day_in_week_emb[dow_token_idx]], dim=-1)

        node_emb = self.node_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, tpn, -1, -1)
        input_data = torch.cat([input_data, node_emb], dim=-1)
        return input_data

    def forward(self, history_data, label=None):
        del label
        if history_data.shape[1] != self.seq_len:
            raise ValueError(
                "Seq len mismatch in PatchSTG: got {}, expected {}".format(
                    history_data.shape[1], self.seq_len
                )
            )
        if history_data.shape[2] != self.node_num:
            raise ValueError(
                "Node num mismatch in PatchSTG: got {}, expected {}".format(
                    history_data.shape[2], self.node_num
                )
            )

        embedded_x = self.embedding(history_data)
        reordered = embedded_x[:, :, self.reo_all_idx, :]

        for block in self.spa_encoder:
            reordered = block(reordered)

        original = torch.zeros(
            reordered.shape[0],
            reordered.shape[1],
            self.node_num,
            reordered.shape[-1],
            device=history_data.device,
            dtype=reordered.dtype,
        )
        original[:, :, self.ori_parts_idx, :] = reordered[:, :, self.reo_parts_idx, :]

        prediction = self.regression_conv(
            original.transpose(2, 3).reshape(original.shape[0], -1, original.shape[-2], 1)
        )
        return prediction.permute(0, 1, 2, 3).contiguous()
