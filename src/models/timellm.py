import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base.model import BaseModel


class TimeLLM(BaseModel):
    """
    LargeST-compatible Time-LLM adaptation.

    The official Time-LLM reprograms time-series patches into LLM token space and
    prepends prompt tokens. In this repository, we keep the core mechanism while
    replacing heavyweight external LLM dependencies with a lightweight Transformer
    backbone so the model can be trained directly inside the existing framework.
    """

    def __init__(
        self,
        traffic_dim,
        d_model,
        llm_dim,
        llm_layers,
        n_heads,
        d_ff,
        patch_len,
        stride,
        prompt_len,
        num_prototypes,
        top_k_lags,
        freeze_backbone,
        use_revin,
        dropout,
        **args
    ):
        super(TimeLLM, self).__init__(**args)

        if self.output_dim != 1:
            raise ValueError(
                "TimeLLM in this LargeST adaptation expects output_dim=1, got {}".format(
                    self.output_dim
                )
            )
        if traffic_dim <= 0 or traffic_dim > self.input_dim:
            raise ValueError(
                "traffic_dim must be in (0, input_dim], got {} with input_dim={}".format(
                    traffic_dim, self.input_dim
                )
            )
        if patch_len <= 0 or stride <= 0:
            raise ValueError("patch_len and stride must be > 0")
        if llm_dim % n_heads != 0:
            raise ValueError("llm_dim must be divisible by n_heads")

        self.traffic_dim = traffic_dim
        self.d_model = d_model
        self.llm_dim = llm_dim
        self.patch_len = patch_len
        self.stride = stride
        self.prompt_len = prompt_len
        self.top_k_lags = max(1, top_k_lags)
        self.use_revin = use_revin
        self.dropout = nn.Dropout(dropout)

        self.patch_proj = nn.Linear(self.patch_len * self.traffic_dim, self.d_model)
        self.patch_norm = nn.LayerNorm(self.d_model)

        # Numeric prompt descriptor (instead of text tokenization).
        stats_dim = 8
        self.stats_proj = nn.Sequential(
            nn.Linear(stats_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        self.prompt_base = nn.Parameter(torch.randn(1, self.prompt_len, self.d_model) * 0.02)
        self.prompt_to_llm = nn.Linear(self.d_model, self.llm_dim)

        self.prototype_embeddings = nn.Parameter(
            torch.randn(num_prototypes, self.llm_dim) * 0.02
        )
        self.reprogramming_layer = ReprogrammingLayer(
            d_model=self.d_model,
            n_heads=n_heads,
            d_keys=self.d_model // n_heads,
            d_llm=self.llm_dim,
            attention_dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.llm_dim,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.llm_backbone = nn.TransformerEncoder(encoder_layer, num_layers=llm_layers)
        self.backbone_norm = nn.LayerNorm(self.llm_dim)

        # Horizon query tokens decode future steps from patch-token memory.
        self.horizon_query = nn.Parameter(torch.randn(1, self.horizon, self.llm_dim) * 0.02)
        self.output_proj = nn.Sequential(
            nn.Linear(self.llm_dim, self.llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.llm_dim, self.output_dim),
        )

        if freeze_backbone:
            for parameter in self.llm_backbone.parameters():
                parameter.requires_grad = False

    def _normalize(self, inputs):
        if not self.use_revin:
            batch_size, _, node_num, feat_dim = inputs.shape
            mean = inputs.new_zeros(batch_size, 1, node_num, feat_dim)
            std = inputs.new_ones(batch_size, 1, node_num, feat_dim)
            return inputs, mean, std

        mean = inputs.mean(dim=1, keepdim=True)
        std = inputs.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        return (inputs - mean) / std, mean, std

    def _patchify(self, inputs):
        # inputs: [B, T, N, F]
        batch_size, _, node_num, _ = inputs.shape
        node_series = inputs.permute(0, 2, 1, 3).contiguous()  # [B, N, T, F]

        pad_len = self.stride
        if pad_len > 0:
            pad = node_series[:, :, -1:, :].expand(-1, -1, pad_len, -1)
            node_series = torch.cat([node_series, pad], dim=2)

        # [B, N, P, F, patch_len] -> [B, N, P, patch_len, F]
        patches = node_series.unfold(dimension=2, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 2, 4, 3).contiguous()
        patch_num = patches.shape[2]
        patches = patches.view(batch_size * node_num, patch_num, self.patch_len * self.traffic_dim)

        patch_tokens = self.patch_proj(patches)
        patch_tokens = self.patch_norm(patch_tokens)
        return patch_tokens, patch_num

    def _dominant_period(self, series):
        # series: [B*N, T]
        centered = series - series.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(centered, dim=-1).abs()
        if spectrum.shape[-1] <= 1:
            period = torch.ones(series.shape[0], device=series.device, dtype=series.dtype)
            strength = torch.zeros_like(period)
            return period, strength

        spectrum = spectrum[:, 1:]
        peak_idx = spectrum.argmax(dim=-1) + 1
        period = float(series.shape[-1]) / peak_idx.float()
        strength = spectrum.max(dim=-1).values / (spectrum.mean(dim=-1) + 1e-6)
        return period, strength

    def _top_lag_feature(self, series):
        # series: [B*N, T]
        max_lag = min(series.shape[-1] - 1, self.top_k_lags * 4)
        if max_lag <= 0:
            return torch.zeros(series.shape[0], device=series.device, dtype=series.dtype)

        centered = series - series.mean(dim=-1, keepdim=True)
        corr_scores = []
        for lag in range(1, max_lag + 1):
            left = centered[:, :-lag]
            right = centered[:, lag:]
            score = (left * right).mean(dim=-1)
            corr_scores.append(score)
        corr = torch.stack(corr_scores, dim=-1)
        topk = torch.topk(corr, k=min(self.top_k_lags, corr.shape[-1]), dim=-1).values
        return topk.mean(dim=-1)

    def _build_prompt_stats(self, inputs):
        # inputs: [B, T, N, F]
        batch_size, seq_len, node_num, _ = inputs.shape
        node_series = inputs.mean(dim=-1)  # [B, T, N]

        min_v = node_series.min(dim=1).values
        max_v = node_series.max(dim=1).values
        median_v = node_series.median(dim=1).values
        mean_v = node_series.mean(dim=1)
        std_v = node_series.std(dim=1, unbiased=False)
        trend = node_series[:, -1, :] - node_series[:, 0, :]

        flattened = node_series.permute(0, 2, 1).reshape(batch_size * node_num, seq_len)
        period, strength = self._dominant_period(flattened)
        lag_score = self._top_lag_feature(flattened)
        period = period.view(batch_size, node_num)
        strength = strength.view(batch_size, node_num)
        lag_score = lag_score.view(batch_size, node_num)

        stats = torch.stack(
            [min_v, max_v, median_v, mean_v, std_v, trend, period, strength + lag_score],
            dim=-1,
        )  # [B, N, 8]
        return stats

    def _prompt_tokens(self, stats):
        # stats: [B, N, S]
        batch_size, node_num, _ = stats.shape
        stats_embed = self.stats_proj(stats.view(batch_size * node_num, -1)).unsqueeze(1)
        prompt_tokens = self.prompt_base.expand(batch_size * node_num, -1, -1) + stats_embed
        return prompt_tokens

    def _positional_encoding(self, token_len, dim, device, dtype):
        position = torch.arange(token_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(token_len, dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, inputs, label=None):
        traffic_inputs = inputs[..., : self.traffic_dim]
        batch_size, _, node_num, _ = traffic_inputs.shape

        norm_inputs, mean, std = self._normalize(traffic_inputs)

        stats = self._build_prompt_stats(norm_inputs)
        prompt_tokens = self._prompt_tokens(stats)
        prompt_tokens = self.prompt_to_llm(prompt_tokens)

        patch_tokens, patch_num = self._patchify(norm_inputs)
        reprog_tokens = self.reprogramming_layer(
            patch_tokens,
            self.prototype_embeddings,
            self.prototype_embeddings,
        )

        tokens = torch.cat([prompt_tokens, reprog_tokens], dim=1)
        tokens = tokens + self._positional_encoding(
            tokens.shape[1], self.llm_dim, tokens.device, tokens.dtype
        )
        tokens = self.dropout(tokens)
        hidden = self.llm_backbone(tokens)
        hidden = self.backbone_norm(hidden)

        patch_hidden = hidden[:, -patch_num:, :]  # [B*N, P, D]

        horizon_query = self.horizon_query.expand(batch_size * node_num, -1, -1)
        attn_scores = torch.matmul(horizon_query, patch_hidden.transpose(1, 2)) / math.sqrt(
            self.llm_dim
        )
        attn = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn, patch_hidden)  # [B*N, H, D]

        prediction = self.output_proj(context)  # [B*N, H, 1]
        prediction = prediction.view(batch_size, node_num, self.horizon, self.output_dim)
        prediction = prediction.permute(0, 2, 1, 3).contiguous()

        if self.use_revin:
            target_mean = mean[..., : self.output_dim]
            target_std = std[..., : self.output_dim]
            prediction = prediction * target_std + target_mean

        return prediction


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys, d_llm, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        # target_embedding: [B, L, d_model]
        # source/value_embedding: [S, d_llm]
        batch_size, seq_len, _ = target_embedding.shape
        source_len, _ = source_embedding.shape
        head_num = self.n_heads

        query = self.query_projection(target_embedding).view(batch_size, seq_len, head_num, -1)
        key = self.key_projection(source_embedding).view(source_len, head_num, -1)
        value = self.value_projection(value_embedding).view(source_len, head_num, -1)

        scale = 1.0 / math.sqrt(query.shape[-1])
        scores = torch.einsum("blhe,she->bhls", query, key)
        attn = self.dropout(torch.softmax(scores * scale, dim=-1))
        reprogrammed = torch.einsum("bhls,she->blhe", attn, value)
        reprogrammed = reprogrammed.reshape(batch_size, seq_len, -1)
        return self.out_projection(reprogrammed)
