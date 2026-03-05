import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function

from src.base.model import BaseModel


class STEVE(BaseModel):
    """
    LargeST-compatible adaptation of:
    Seeing the Unseen: Learning Basis Confounder Representations for Robust Traffic Prediction.

    This implementation keeps the key ideas:
    - dual ST encoders (variant / invariant)
    - adaptive graph for the variant branch
    - basis-bank confounder extraction
    - auxiliary variant / invariant objectives
    - mutual-information regularization

    Input layout:
    - first `traffic_dim` channels are traffic features
    """

    def __init__(
        self,
        base_adj,
        traffic_dim,
        embed_dim,
        ks,
        kt,
        st_dropout,
        bank_ratio,
        bank_gamma,
        temporal_classes,
        congestion_channel,
        spatial_sample_size,
        mi_sample_size,
        grl_alpha,
        use_grl,
        **args
    ):
        super(STEVE, self).__init__(**args)

        if traffic_dim <= 0:
            raise ValueError("traffic_dim must be > 0")
        if traffic_dim > self.input_dim:
            raise ValueError(
                "traffic_dim must be <= input_dim, got {} > {}".format(traffic_dim, self.input_dim)
            )
        if self.seq_len <= 4 * (kt - 1):
            raise ValueError(
                "seq_len must be > 4*(kt-1) for two ST blocks, got seq_len={} kt={}".format(
                    self.seq_len, kt
                )
            )
        if base_adj.shape[0] != self.node_num or base_adj.shape[1] != self.node_num:
            raise ValueError(
                "base_adj shape mismatch, expected ({}, {}), got {}".format(
                    self.node_num, self.node_num, tuple(base_adj.shape)
                )
            )

        self.traffic_dim = traffic_dim
        self.embed_dim = embed_dim
        self.ks = ks
        self.kt = kt
        self.bank_gamma = bank_gamma
        self.temporal_classes = max(2, temporal_classes)
        self.congestion_channel = max(0, min(congestion_channel, traffic_dim - 1))
        self.spatial_sample_size = max(1, spatial_sample_size)
        self.mi_sample_size = max(1, mi_sample_size)
        self.use_grl = use_grl

        self.repr_len = self.seq_len - 4 * (self.kt - 1)

        self.register_buffer("base_adj", base_adj.float())
        self.register_buffer("spatial_targets", torch.arange(self.node_num, dtype=torch.long))

        self.variant_encoder = STEncoder(
            num_nodes=self.node_num,
            d_input=self.traffic_dim,
            d_output=self.embed_dim,
            ks=self.ks,
            kt=self.kt,
            drop_prob=st_dropout,
            input_window=self.seq_len,
        )
        self.invariant_encoder = STEncoder(
            num_nodes=self.node_num,
            d_input=self.traffic_dim,
            d_output=self.embed_dim,
            ks=self.ks,
            kt=self.kt,
            drop_prob=st_dropout,
            input_window=self.seq_len,
        )

        self.node_embeddings_1 = nn.Parameter(torch.randn(self.ks, self.node_num, self.embed_dim))
        self.node_embeddings_2 = nn.Parameter(torch.randn(self.ks, self.embed_dim, self.node_num))

        self.temporal_conv_variant = nn.Conv2d(self.repr_len, self.horizon, kernel_size=1, bias=True)
        self.temporal_conv_invariant = nn.Conv2d(self.repr_len, self.horizon, kernel_size=1, bias=True)

        self.variant_predict = nn.Linear(self.embed_dim, self.output_dim)
        self.invariant_predict = nn.Linear(self.embed_dim, self.output_dim)
        self.c_weight = nn.Linear(self.embed_dim, self.output_dim)

        bank_size = max(8, int(self.embed_dim * bank_ratio))
        bank_init = F.normalize(torch.randn(bank_size, self.embed_dim), dim=-1)
        self.register_buffer("bank", bank_init)
        self.bank_proj = nn.Linear(self.repr_len * self.node_num, bank_size)
        self.bank_attn = BankAttention(self.embed_dim)

        self.variant_temporal_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, self.temporal_classes),
        )
        self.variant_congest_head = nn.Sequential(
            nn.Linear(self.embed_dim, max(2, self.embed_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(2, self.embed_dim // 2), 1),
        )
        self.variant_node_prototypes = nn.Parameter(torch.randn(self.node_num, self.embed_dim))

        self.invariant_temporal_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, self.temporal_classes),
        )
        self.invariant_congest_head = nn.Sequential(
            nn.Linear(self.embed_dim, max(2, self.embed_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(2, self.embed_dim // 2), 1),
        )
        self.invariant_node_prototypes = nn.Parameter(torch.randn(self.node_num, self.embed_dim))

        self.revgrad = RevGradLayer(alpha=grl_alpha)
        self.mi_net = CLUB(self.embed_dim, self.embed_dim, hidden_size=max(64, self.embed_dim * 4))

        self._reset_parameters()

    def _reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.uniform_(parameter)

    def _adaptive_graph(self):
        adaptive_adj = torch.einsum("knd,kdm->knm", self.node_embeddings_1, self.node_embeddings_2)
        adaptive_adj = torch.softmax(F.relu(adaptive_adj), dim=-1)
        return adaptive_adj

    def _confounder_ext(self, z_tensor):
        # z_tensor: [B, T_repr, N, D]
        batch_size, t_repr, node_num, emb_dim = z_tensor.shape
        z_flat = z_tensor.reshape(batch_size, t_repr * node_num, emb_dim).permute(0, 2, 1)  # [B, D, T*N]
        bank_candidates = self.bank_proj(z_flat).permute(0, 2, 1)  # [B, K, D]

        if self.training:
            with torch.no_grad():
                self.bank.copy_(self.bank_gamma * self.bank + (1.0 - self.bank_gamma) * bank_candidates.mean(dim=0))

        shared_bank = self.bank.unsqueeze(0).expand(batch_size, -1, -1)
        mixed_bank = self.bank_gamma * shared_bank + (1.0 - self.bank_gamma) * bank_candidates
        query = z_tensor.mean(dim=1)  # [B, N, D]
        confounder, attn = self.bank_attn(query, mixed_bank, mixed_bank)
        return confounder, attn

    def _sample_nodes(self, device):
        sample_size = min(self.node_num, self.spatial_sample_size)
        if sample_size == self.node_num:
            return self.spatial_targets.to(device)
        if self.training:
            return torch.randperm(self.node_num, device=device)[:sample_size]
        return self.spatial_targets[:sample_size].to(device)

    def _spatial_loss(self, node_repr, prototypes):
        sampled_nodes = self._sample_nodes(node_repr.device)
        sampled_repr = F.normalize(node_repr[:, sampled_nodes, :].mean(dim=0), dim=-1)
        sampled_proto = F.normalize(prototypes, dim=-1)
        logits = torch.matmul(sampled_repr, sampled_proto.t())
        return F.cross_entropy(logits, sampled_nodes)

    def _build_aux_targets(self, traffic_inputs):
        # temporal pseudo label from trend
        trend_series = traffic_inputs.mean(dim=(2, 3))  # [B, T]
        trend = trend_series[:, -1] - trend_series[:, 0]
        trend = (trend - trend.mean()) / (trend.std(unbiased=False) + 1e-6)
        trend = torch.sigmoid(trend)
        temporal_target = torch.clamp(
            (trend * self.temporal_classes).long(),
            min=0,
            max=self.temporal_classes - 1,
        )

        # congestion pseudo target from selected channel
        congestion = traffic_inputs[..., self.congestion_channel].mean(dim=1)  # [B, N]
        c_min = congestion.amin(dim=1, keepdim=True)
        c_max = congestion.amax(dim=1, keepdim=True)
        congestion_target = (congestion - c_min) / (c_max - c_min + 1e-6)
        return temporal_target, congestion_target.unsqueeze(-1)

    def _variant_aux_loss(self, confounder_repr, temporal_target, congestion_target):
        temporal_repr = confounder_repr.mean(dim=1)  # [B, D]
        temporal_logits = self.variant_temporal_head(temporal_repr)
        temporal_loss = F.cross_entropy(temporal_logits, temporal_target)

        spatial_loss = self._spatial_loss(confounder_repr, self.variant_node_prototypes)

        congestion_pred = self.variant_congest_head(confounder_repr)
        congestion_loss = F.mse_loss(congestion_pred, congestion_target)

        return (temporal_loss + spatial_loss + congestion_loss) / 3.0

    def _invariant_aux_loss(self, invariant_repr, temporal_target, congestion_target, progress):
        if self.training and self.use_grl:
            invariant_repr = self.revgrad(invariant_repr, progress)

        temporal_repr = invariant_repr.mean(dim=1)  # [B, D]
        temporal_logits = self.invariant_temporal_head(temporal_repr)
        temporal_loss = F.cross_entropy(temporal_logits, temporal_target)

        spatial_loss = self._spatial_loss(invariant_repr, self.invariant_node_prototypes)

        congestion_pred = self.invariant_congest_head(invariant_repr)
        congestion_loss = F.mse_loss(congestion_pred, congestion_target)

        return (temporal_loss + spatial_loss + congestion_loss) / 3.0

    def _mi_regularizer(self, invariant_repr, confounder_repr):
        h_flat = invariant_repr.reshape(-1, invariant_repr.shape[-1])
        c_flat = confounder_repr.reshape(-1, confounder_repr.shape[-1])
        if h_flat.shape[0] > self.mi_sample_size:
            sample_idx = torch.randperm(h_flat.shape[0], device=h_flat.device)[: self.mi_sample_size]
            h_flat = h_flat[sample_idx]
            c_flat = c_flat[sample_idx]
        return self.mi_net(h_flat, c_flat)

    def forward(self, inputs, label=None, progress=0.0, training=False):
        traffic_inputs = inputs[..., : self.traffic_dim]
        x = traffic_inputs.permute(0, 3, 1, 2)  # [B, F, T, N]

        invariant_feature = self.invariant_encoder(x, self.base_adj)  # [B, D, T_repr, N]
        variant_feature = self.variant_encoder.variant_encode(x, self._adaptive_graph())

        h_tensor = invariant_feature.permute(0, 2, 3, 1)  # [B, T_repr, N, D]
        z_tensor = variant_feature.permute(0, 2, 3, 1)

        confounder_repr, _ = self._confounder_ext(z_tensor)  # [B, N, D]
        c_expand = confounder_repr.unsqueeze(1).expand(-1, self.horizon, -1, -1)

        h_projected = self.temporal_conv_invariant(h_tensor)  # [B, H, N, D]
        z_projected = self.temporal_conv_variant(z_tensor)  # [B, H, N, D]
        variant_state = z_projected + c_expand

        y_c = self.variant_predict(variant_state)
        y_h = self.invariant_predict(h_projected)
        c_weight = torch.relu(self.c_weight(c_expand))
        prediction = y_h + c_weight * y_c

        outputs = {"prediction": prediction}
        if training:
            temporal_target, congestion_target = self._build_aux_targets(traffic_inputs)
            invariant_repr = h_projected.mean(dim=1)  # [B, N, D]
            variant_loss = self._variant_aux_loss(confounder_repr, temporal_target, congestion_target)
            invariant_loss = self._invariant_aux_loss(
                invariant_repr, temporal_target, congestion_target, progress
            )
            mi_loss = self._mi_regularizer(invariant_repr, confounder_repr)
            outputs["variant_loss"] = variant_loss
            outputs["invariant_loss"] = invariant_loss
            outputs["mi_loss"] = mi_loss
        else:
            zero = prediction.new_tensor(0.0)
            outputs["variant_loss"] = zero
            outputs["invariant_loss"] = zero
            outputs["mi_loss"] = zero

        return outputs


class RevGradFunc(Function):
    @staticmethod
    def forward(ctx, input_tensor, alpha):
        ctx.save_for_backward(alpha)
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        (alpha,) = ctx.saved_tensors
        return -grad_output * alpha, None


class RevGradLayer(nn.Module):
    def __init__(self, alpha=0.01):
        super(RevGradLayer, self).__init__()
        self.register_buffer("_base_alpha", torch.tensor(float(alpha)))

    def forward(self, inputs, progress):
        alpha = self._base_alpha / math.pow(1.0 + 10.0 * float(progress), 0.75)
        alpha = torch.tensor(alpha, dtype=inputs.dtype, device=inputs.device)
        return RevGradFunc.apply(inputs, alpha)


class CLUB(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        hidden_size = max(16, hidden_size)
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
            nn.Tanh(),
        )

    def forward(self, x_samples, y_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples).clamp(min=-5.0, max=5.0)

        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size, device=x_samples.device)

        positive = -((mu - y_samples) ** 2) / logvar.exp()
        negative = -((mu - y_samples[random_index]) ** 2) / logvar.exp()
        return 0.5 * (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()


class STEncoder(nn.Module):
    def __init__(self, num_nodes, d_input, d_output, ks, kt, drop_prob, input_window):
        super(STEncoder, self).__init__()
        if input_window - 4 * (kt - 1) <= 0:
            raise ValueError("input_window must be > 4*(kt-1)")

        blocks = [
            [d_output, d_output // 2, d_output],
            [d_output, d_output // 2, d_output],
        ]
        self.ks = ks
        self.input_conv = nn.Conv2d(d_input, d_output, kernel_size=1)
        self.st_conv1 = STConvBlock(ks, kt, num_nodes, blocks[0], drop_prob)
        self.st_conv2 = STConvBlock(ks, kt, num_nodes, blocks[1], drop_prob)

    def _graph_basis(self, graph):
        if graph.dim() == 2:
            lap_mx = cal_laplacian(graph)
            return cal_cheb_polynomial(lap_mx, self.ks)
        if graph.dim() == 3:
            if graph.shape[0] != self.ks:
                raise ValueError(
                    "dynamic graph first dimension must equal ks, got {} vs {}".format(
                        graph.shape[0], self.ks
                    )
                )
            return graph
        raise ValueError("graph must be 2D or 3D tensor")

    def forward(self, x, graph):
        x = self.input_conv(x)
        graph_basis = self._graph_basis(graph)
        x = self.st_conv1(x, graph_basis)
        x = self.st_conv2(x, graph_basis)
        return x

    def variant_encode(self, x, graph):
        return self.forward(x, graph)


def cal_laplacian(graph):
    identity = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype)
    graph = graph + identity
    degree = torch.diag(torch.sum(graph, dim=-1).clamp(min=1e-6).pow(-0.5))
    laplacian = identity - torch.mm(torch.mm(degree, graph), degree)
    return laplacian


def cal_cheb_polynomial(laplacian, order):
    node_num = laplacian.size(0)
    cheb = torch.zeros((order, node_num, node_num), device=laplacian.device, dtype=laplacian.dtype)
    cheb[0] = torch.eye(node_num, device=laplacian.device, dtype=laplacian.dtype)
    if order == 1:
        return cheb
    cheb[1] = laplacian
    for k in range(2, order):
        cheb[k] = 2 * torch.mm(laplacian, cheb[k - 1]) - cheb[k - 2]
    return cheb


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.conv1x1 = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in > c_out else None

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            pad_channels = self.c_out - self.c_in
            return F.pad(x, [0, 0, 0, 0, 0, pad_channels, 0, 0])
        return x


class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, kernel_size=(kt, 1), stride=1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=(kt, 1), stride=1)

    def forward(self, x):
        x_in = self.align(x)[:, :, self.kt - 1 :, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, : self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out :, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)


class SpatioConvLayer(nn.Module):
    def __init__(self, ks, c_in, c_out):
        super(SpatioConvLayer, self).__init__()
        self.theta = nn.Parameter(torch.empty(c_in, c_out, ks))
        self.bias = nn.Parameter(torch.empty(1, c_out, 1, 1))
        self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1.0 / math.sqrt(max(1, fan_in))
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x, graph_basis):
        x_c = torch.einsum("knm,bitm->bitkn", graph_basis, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.bias
        x_in = self.align(x)
        return torch.relu(x_gc + x_in)


class STConvBlock(nn.Module):
    def __init__(self, ks, kt, node_num, channels, dropout):
        super(STConvBlock, self).__init__()
        self.tconv1 = TemporalConvLayer(kt, channels[0], channels[1], act="GLU")
        self.sconv = SpatioConvLayer(ks, channels[1], channels[1])
        self.tconv2 = TemporalConvLayer(kt, channels[1], channels[2], act="relu")
        self.layer_norm = nn.LayerNorm([node_num, channels[2]])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, graph_basis):
        x = self.tconv1(x)
        x = self.sconv(x, graph_basis)
        x = self.tconv2(x)
        x = self.layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x)


class BankAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BankAttention, self).__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        scale = 1.0 / math.sqrt(q.shape[-1])
        attn = torch.softmax(torch.matmul(q, k.transpose(1, 2)) * scale, dim=-1)
        output = torch.matmul(attn, v)
        return output, attn
