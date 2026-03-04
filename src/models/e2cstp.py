import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base.model import BaseModel


class E2CSTP(BaseModel):
    """
    A LargeST-friendly implementation of E^2-CSTP.

    The original paper is multi-modal. In this repository we keep the existing
    data format and split the input channels into three aligned modalities:
    spatio-temporal, text, and image. Missing modalities can be disabled by
    setting their dimensions to 0.
    """

    def __init__(
        self,
        supports,
        st_dim,
        text_dim,
        image_dim,
        hidden_dim,
        encoder_layers,
        num_heads,
        mamba_expand,
        mamba_kernel,
        gcn_order,
        decoder_hidden,
        graph_fusion,
        causal_momentum,
        causal_update_interval,
        intervention_scale,
        dropout,
        **args
    ):
        super(E2CSTP, self).__init__(**args)

        if st_dim + text_dim + image_dim != self.input_dim:
            raise ValueError(
                "st_dim + text_dim + image_dim must equal input_dim, got "
                "{} + {} + {} != {}".format(st_dim, text_dim, image_dim, self.input_dim)
            )

        self.st_dim = st_dim
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.graph_fusion = graph_fusion
        self.causal_momentum = causal_momentum
        self.causal_update_interval = max(1, causal_update_interval)
        self.intervention_scale = intervention_scale
        self._causal_step = 0

        for idx, support in enumerate(supports):
            self.register_buffer("support_{}".format(idx), support.float())
        self.support_len = len(supports)
        self.register_buffer("causal_importance", torch.ones(self.node_num))

        self.st_proj = nn.Linear(self.st_dim, hidden_dim)
        self.text_proj = nn.Linear(self.text_dim, hidden_dim) if self.text_dim > 0 else None
        self.image_proj = nn.Linear(self.image_dim, hidden_dim) if self.image_dim > 0 else None

        self.text_missing = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))
        self.image_missing = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

        self.text_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.image_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.modal_gate = nn.Linear(hidden_dim * 3, 3)
        self.modal_norm = nn.LayerNorm(hidden_dim)

        self.confounder_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.intervention_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.intervention_norm = nn.LayerNorm(hidden_dim)

        self.causal_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.encoder = SpatioTemporalEncoder(
            hidden_dim=hidden_dim,
            layers=encoder_layers,
            support_len=self.support_len,
            gcn_order=gcn_order,
            mamba_expand=mamba_expand,
            mamba_kernel=mamba_kernel,
            dropout=dropout,
        )
        self.feature_mixer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.prediction_head = PredictionHead(
            seq_len=self.seq_len,
            hidden_dim=hidden_dim,
            decoder_hidden=decoder_hidden,
            horizon=self.horizon,
            output_dim=self.output_dim,
            dropout=dropout,
        )

        nn.init.constant_(self.modal_gate.bias, 0.0)
        with torch.no_grad():
            self.modal_gate.bias[0] = 1.0

    def _supports(self):
        return [getattr(self, "support_{}".format(idx)) for idx in range(self.support_len)]

    def _split_modalities(self, inputs):
        cursor = 0
        st_inputs = inputs[..., cursor : cursor + self.st_dim]
        cursor += self.st_dim

        text_inputs = None
        if self.text_dim > 0:
            text_inputs = inputs[..., cursor : cursor + self.text_dim]
            cursor += self.text_dim

        image_inputs = None
        if self.image_dim > 0:
            image_inputs = inputs[..., cursor : cursor + self.image_dim]

        return st_inputs, text_inputs, image_inputs

    def _encode_optional(self, inputs, projector, missing_token, reference):
        if inputs is None:
            return missing_token.expand(reference.shape[0], reference.shape[1], reference.shape[2], -1)
        return projector(inputs)

    def _update_causal_importance(self, fused_hidden):
        if not self.training:
            return

        self._causal_step += 1
        if self._causal_step % self.causal_update_interval != 0:
            return

        with torch.no_grad():
            node_summary = fused_hidden.detach().mean(dim=(0, 1))
            node_importance = torch.sigmoid(self.causal_scorer(node_summary)).squeeze(-1)
            self.causal_importance.mul_(self.causal_momentum).add_(
                node_importance * (1.0 - self.causal_momentum)
            )

    def _hybrid_supports(self):
        importance = self.causal_importance.clamp(0.0, 1.0)
        row_scale = importance.view(-1, 1)
        col_scale = importance.view(1, -1)

        hybrid_supports = []
        for support in self._supports():
            causal_support = support * row_scale * col_scale
            hybrid_supports.append(
                self.graph_fusion * support + (1.0 - self.graph_fusion) * causal_support
            )
        return hybrid_supports

    def forward(self, inputs, label=None):
        st_inputs, text_inputs, image_inputs = self._split_modalities(inputs)

        st_hidden = self.st_proj(st_inputs)
        text_hidden = self._encode_optional(text_inputs, self.text_proj, self.text_missing, st_hidden)
        image_hidden = self._encode_optional(image_inputs, self.image_proj, self.image_missing, st_hidden)

        text_context = self.text_attn(st_hidden, text_hidden)
        image_context = self.image_attn(st_hidden, image_hidden)

        gate = torch.softmax(
            self.modal_gate(torch.cat([st_hidden, text_context, image_context], dim=-1)),
            dim=-1,
        )
        fused_hidden = (
            gate[..., 0:1] * st_hidden
            + gate[..., 1:2] * text_context
            + gate[..., 2:3] * image_context
        )
        fused_hidden = self.modal_norm(fused_hidden + st_hidden)

        confounder = torch.tanh(self.confounder_proj(torch.cat([text_hidden, image_hidden], dim=-1)))
        intervention = torch.sigmoid(
            self.intervention_gate(torch.cat([fused_hidden, confounder], dim=-1))
        )
        adjusted_hidden = self.intervention_norm(
            fused_hidden - self.intervention_scale * intervention * confounder
        )

        adjusted_centered = adjusted_hidden - adjusted_hidden.mean(dim=(0, 1, 2), keepdim=True)
        confounder_centered = confounder - confounder.mean(dim=(0, 1, 2), keepdim=True)
        causal_penalty = (adjusted_centered * confounder_centered).mean(dim=(0, 1, 2)).abs().mean()

        self._update_causal_importance(fused_hidden)
        hybrid_supports = self._hybrid_supports()

        main_hidden = self.encoder(st_hidden, hybrid_supports)
        aux_hidden = self.encoder(adjusted_hidden, hybrid_supports)
        fused_output = self.feature_mixer(torch.cat([main_hidden, aux_hidden], dim=-1))

        main_pred = self.prediction_head(main_hidden)
        aux_pred = self.prediction_head(aux_hidden)
        final_pred = self.prediction_head(fused_output)

        return {
            "prediction": final_pred,
            "main_prediction": main_pred,
            "aux_prediction": aux_pred,
            "causal_penalty": causal_penalty,
        }


class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(CrossModalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, context):
        batch_size, seq_len, node_num, hidden_dim = query.shape
        query = query.permute(0, 2, 1, 3).reshape(batch_size * node_num, seq_len, hidden_dim)
        context = context.permute(0, 2, 1, 3).reshape(batch_size * node_num, seq_len, hidden_dim)
        attended, _ = self.attention(query, context, context, need_weights=False)
        attended = self.norm(query + self.dropout(attended))
        return attended.view(batch_size, node_num, seq_len, hidden_dim).permute(0, 2, 1, 3)


class SpatioTemporalEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        layers,
        support_len,
        gcn_order,
        mamba_expand,
        mamba_kernel,
        dropout,
    ):
        super(SpatioTemporalEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                STEDLayer(
                    hidden_dim=hidden_dim,
                    support_len=support_len,
                    gcn_order=gcn_order,
                    mamba_expand=mamba_expand,
                    mamba_kernel=mamba_kernel,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, inputs, supports):
        hidden = inputs
        for layer in self.layers:
            hidden = layer(hidden, supports)
        return hidden


class STEDLayer(nn.Module):
    def __init__(self, hidden_dim, support_len, gcn_order, mamba_expand, mamba_kernel, dropout):
        super(STEDLayer, self).__init__()
        self.spatial = DiffusionGraphConv(
            hidden_dim=hidden_dim,
            support_len=support_len,
            order=gcn_order,
            dropout=dropout,
        )
        self.temporal = TemporalMambaBlock(
            hidden_dim=hidden_dim,
            expand=mamba_expand,
            kernel_size=mamba_kernel,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs, supports):
        spatial_hidden = self.spatial(inputs, supports)
        temporal_hidden = self.temporal(inputs)
        fused_hidden = inputs + spatial_hidden + temporal_hidden + spatial_hidden * temporal_hidden
        return self.norm(fused_hidden)


class DiffusionGraphConv(nn.Module):
    def __init__(self, hidden_dim, support_len, order, dropout):
        super(DiffusionGraphConv, self).__init__()
        self.order = order
        input_dim = hidden_dim * (1 + support_len * order)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _propagate(self, inputs, support):
        return torch.einsum("btnd,nm->btmd", inputs, support)

    def forward(self, inputs, supports):
        features = [inputs]
        for support in supports:
            hidden = self._propagate(inputs, support)
            features.append(hidden)
            for _ in range(1, self.order):
                hidden = self._propagate(hidden, support)
                features.append(hidden)

        hidden = torch.cat(features, dim=-1)
        hidden = F.gelu(self.proj(hidden))
        return self.dropout(hidden)


class TemporalMambaBlock(nn.Module):
    def __init__(self, hidden_dim, expand, kernel_size, dropout):
        super(TemporalMambaBlock, self).__init__()
        inner_dim = hidden_dim * expand
        self.inner_dim = inner_dim
        self.in_proj = nn.Linear(hidden_dim, inner_dim * 2)
        self.depthwise_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=inner_dim,
        )
        self.delta_proj = nn.Linear(inner_dim, inner_dim)
        self.state_in_proj = nn.Linear(inner_dim, inner_dim)
        self.state_out_proj = nn.Linear(inner_dim, inner_dim)
        self.a_log = nn.Parameter(torch.zeros(inner_dim))
        self.d_skip = nn.Parameter(torch.ones(inner_dim))
        self.out_proj = nn.Linear(inner_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _selective_scan(self, inputs):
        batch_size, seq_len, hidden_dim = inputs.shape
        delta = F.softplus(self.delta_proj(inputs))
        state_in = torch.tanh(self.state_in_proj(inputs))
        state_out = torch.tanh(self.state_out_proj(inputs))

        state = torch.zeros(batch_size, hidden_dim, device=inputs.device, dtype=inputs.dtype)
        a = -F.softplus(self.a_log).view(1, hidden_dim)
        d = self.d_skip.view(1, hidden_dim)

        outputs = []
        for step in range(seq_len):
            decay = torch.exp(delta[:, step, :] * a)
            state = decay * state + state_in[:, step, :] * inputs[:, step, :]
            outputs.append((state_out[:, step, :] * state + d * inputs[:, step, :]).unsqueeze(1))
        return torch.cat(outputs, dim=1)

    def forward(self, inputs):
        batch_size, seq_len, node_num, hidden_dim = inputs.shape
        inputs = inputs.permute(0, 2, 1, 3).reshape(batch_size * node_num, seq_len, hidden_dim)
        signal, gate = self.in_proj(inputs).chunk(2, dim=-1)
        signal = signal.transpose(1, 2)
        signal = self.depthwise_conv(signal)[..., :seq_len].transpose(1, 2)
        signal = F.silu(signal)
        hidden = self._selective_scan(signal)
        hidden = hidden * torch.sigmoid(gate)
        hidden = self.dropout(self.out_proj(hidden))
        return hidden.view(batch_size, node_num, seq_len, hidden_dim).permute(0, 2, 1, 3)


class PredictionHead(nn.Module):
    def __init__(self, seq_len, hidden_dim, decoder_hidden, horizon, output_dim, dropout):
        super(PredictionHead, self).__init__()
        self.horizon = horizon
        self.output_dim = output_dim
        self.proj = nn.Sequential(
            nn.Linear(seq_len * hidden_dim, decoder_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, horizon * output_dim),
        )

    def forward(self, inputs):
        batch_size, seq_len, node_num, hidden_dim = inputs.shape
        hidden = inputs.permute(0, 2, 1, 3).reshape(batch_size, node_num, seq_len * hidden_dim)
        hidden = self.proj(hidden)
        hidden = hidden.view(batch_size, node_num, self.horizon, self.output_dim)
        return hidden.permute(0, 2, 1, 3).contiguous()
