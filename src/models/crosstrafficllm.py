import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base.model import BaseModel


class CrossTrafficLLM(BaseModel):
    """
    LargeST-compatible CrossTrafficLLM adaptation.

    Input layout:
    - first `traffic_dim` channels: traffic history features
    - next `text_dim` channels: aligned text embeddings per timestamp-node pair

    Optional report generation is enabled when `report_vocab_size > 0` and
    sidecar files `report_{train,val,test}.npy` are available.
    """

    def __init__(
        self,
        supports,
        traffic_dim,
        text_dim,
        hidden_dim,
        text_hidden,
        num_heads,
        num_layers,
        ff_dim,
        graph_order,
        patch_len,
        top_k_text,
        graph_fusion,
        anomaly_topk,
        report_len,
        report_vocab_size,
        memory_size,
        dropout,
        **args
    ):
        super(CrossTrafficLLM, self).__init__(**args)

        if traffic_dim + text_dim != self.input_dim:
            raise ValueError(
                "traffic_dim + text_dim must equal input_dim, got {} + {} != {}".format(
                    traffic_dim, text_dim, self.input_dim
                )
            )
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if text_dim > 0 and text_hidden % num_heads != 0:
            raise ValueError("text_hidden must be divisible by num_heads when text_dim > 0")

        self.traffic_dim = traffic_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.patch_len = patch_len
        self.graph_fusion = graph_fusion

        for idx, support in enumerate(supports):
            self.register_buffer("support_{}".format(idx), support.float())
        self.support_len = len(supports)

        self.traffic_proj = nn.Linear(traffic_dim, hidden_dim)
        self.traffic_norm = nn.LayerNorm(hidden_dim)

        self.text_missing = nn.Parameter(torch.zeros(1, self.seq_len, self.node_num, hidden_dim))
        if text_dim > 0:
            self.text_proj = nn.Linear(text_dim, text_hidden)
            self.text_encoder = LocationAwareTextEncoder(text_hidden=text_hidden, dropout=dropout)
            self.text_to_hidden = nn.Linear(text_hidden, hidden_dim)
        else:
            self.text_proj = None
            self.text_encoder = None
            self.text_to_hidden = None

        self.cross_modal_align = SparseCrossModalAlign(
            hidden_dim=hidden_dim,
            top_k=top_k_text,
            dropout=dropout,
        )
        self.film = nn.Linear(hidden_dim, hidden_dim * 2)

        self.graph_encoder = TextGuidedAdaptiveGraphConv(
            node_num=self.node_num,
            hidden_dim=hidden_dim,
            support_len=self.support_len + 1,
            order=graph_order,
            graph_fusion=graph_fusion,
            dropout=dropout,
        )

        self.crossformer = TextCrossformerEncoder(
            seq_len=self.seq_len,
            patch_len=patch_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(self.crossformer.output_tokens * hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, self.horizon * self.output_dim),
        )

        self.report_generator = None
        if report_vocab_size > 0:
            self.report_generator = RoadReportGenerator(
                hidden_dim=hidden_dim,
                report_len=report_len,
                vocab_size=report_vocab_size,
                memory_size=memory_size,
                anomaly_topk=anomaly_topk,
                num_heads=num_heads,
                dropout=dropout,
            )

    def _supports(self):
        return [getattr(self, "support_{}".format(idx)) for idx in range(self.support_len)]

    def _encode_text(self, text_inputs, traffic_hidden):
        if self.text_dim == 0:
            return self.text_missing.expand(traffic_hidden.shape[0], -1, -1, -1)
        text_hidden = self.text_proj(text_inputs)
        text_hidden = self.text_encoder(text_hidden)
        return self.text_to_hidden(text_hidden)

    def forward(self, inputs, label=None):
        traffic_inputs = inputs[..., : self.traffic_dim]
        text_inputs = inputs[..., self.traffic_dim :]

        traffic_hidden = self.traffic_norm(self.traffic_proj(traffic_inputs))
        text_hidden = self._encode_text(text_inputs, traffic_hidden)

        aligned_text, _ = self.cross_modal_align(traffic_hidden, text_hidden)
        gamma, beta = self.film(aligned_text).chunk(2, dim=-1)
        fused_hidden = traffic_hidden * (1.0 + torch.tanh(gamma)) + beta

        if self.text_dim > 0:
            pooled_traffic = F.normalize(traffic_hidden.mean(dim=1), dim=-1)
            pooled_text = F.normalize(aligned_text.mean(dim=1), dim=-1)
            alignment_loss = 1.0 - (pooled_traffic * pooled_text).sum(dim=-1).mean()
        else:
            alignment_loss = traffic_hidden.new_tensor(0.0)

        graph_hidden = self.graph_encoder(fused_hidden, aligned_text, self._supports())
        encoded_hidden = self.crossformer(graph_hidden, aligned_text)

        batch_size = encoded_hidden.shape[0]
        pred_hidden = encoded_hidden.permute(0, 2, 1, 3).reshape(
            batch_size, self.node_num, self.crossformer.output_tokens * self.hidden_dim
        )
        prediction = self.prediction_head(pred_hidden)
        prediction = prediction.view(batch_size, self.node_num, self.horizon, self.output_dim)
        prediction = prediction.permute(0, 2, 1, 3).contiguous()

        report_logits = None
        road_scores = None
        if self.report_generator is not None:
            report_logits, road_scores = self.report_generator(encoded_hidden)

        return {
            "prediction": prediction,
            "report_logits": report_logits,
            "road_scores": road_scores,
            "alignment_loss": alignment_loss,
        }


class LocationAwareTextEncoder(nn.Module):
    def __init__(self, text_hidden, dropout, kernel_size=3, global_kernel=5):
        super(LocationAwareTextEncoder, self).__init__()
        self.local_conv = nn.Conv1d(
            text_hidden,
            text_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=1,
        )
        self.global_conv = nn.Conv1d(
            text_hidden,
            text_hidden,
            kernel_size=global_kernel,
            padding=global_kernel // 2,
            groups=text_hidden,
        )
        self.gate = nn.Linear(text_hidden * 2, text_hidden)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(text_hidden)

    def forward(self, text_inputs):
        batch_size, seq_len, node_num, text_hidden = text_inputs.shape
        x = text_inputs.permute(0, 2, 3, 1).reshape(batch_size * node_num, text_hidden, seq_len)
        local_feat = self.local_conv(x)
        global_feat = self.global_conv(x)
        fused = torch.cat([local_feat, global_feat], dim=1).transpose(1, 2)
        fused = fused.view(batch_size, node_num, seq_len, text_hidden * 2).permute(0, 2, 1, 3)
        gate = torch.sigmoid(self.gate(fused))
        output = gate * fused[..., :text_hidden] + (1.0 - gate) * fused[..., text_hidden:]
        return self.norm(text_inputs + self.dropout(output))


class SparseCrossModalAlign(nn.Module):
    def __init__(self, hidden_dim, top_k, dropout):
        super(SparseCrossModalAlign, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, traffic_hidden, text_hidden):
        batch_size, seq_len, node_num, hidden_dim = traffic_hidden.shape
        q = self.query_proj(traffic_hidden).permute(0, 2, 1, 3).reshape(batch_size * node_num, seq_len, hidden_dim)
        k = self.key_proj(text_hidden).permute(0, 2, 1, 3).reshape(batch_size * node_num, seq_len, hidden_dim)
        v = self.value_proj(text_hidden).permute(0, 2, 1, 3).reshape(batch_size * node_num, seq_len, hidden_dim)

        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(hidden_dim)
        top_k = min(self.top_k, scores.shape[-1])
        if top_k < scores.shape[-1]:
            topk_values, topk_indices = torch.topk(scores, k=top_k, dim=-1)
            masked_scores = torch.full_like(scores, float("-inf"))
            masked_scores.scatter_(-1, topk_indices, topk_values)
            scores = masked_scores

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        aligned = torch.matmul(attn, v)
        aligned = aligned.view(batch_size, node_num, seq_len, hidden_dim).permute(0, 2, 1, 3)
        return self.norm(traffic_hidden + aligned), attn


class TextGuidedAdaptiveGraphConv(nn.Module):
    def __init__(self, node_num, hidden_dim, support_len, order, graph_fusion, dropout):
        super(TextGuidedAdaptiveGraphConv, self).__init__()
        self.order = order
        self.graph_fusion = graph_fusion
        self.nodevec1 = nn.Parameter(torch.randn(node_num, hidden_dim))
        self.nodevec2 = nn.Parameter(torch.randn(hidden_dim, node_num))
        self.text_gate = nn.Linear(hidden_dim, 1)
        self.proj = nn.Linear(hidden_dim * (1 + support_len * order), hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def _propagate(self, inputs, support):
        if support.dim() == 2:
            return torch.einsum("btnd,nm->btmd", inputs, support)
        return torch.einsum("btnd,bnm->btmd", inputs, support)

    def forward(self, inputs, text_hidden, supports):
        base_adaptive = F.softmax(F.relu(torch.matmul(self.nodevec1, self.nodevec2)), dim=-1)
        node_gate = torch.sigmoid(self.text_gate(text_hidden.mean(dim=1))).squeeze(-1)
        text_adaptive = base_adaptive.unsqueeze(0) * node_gate.unsqueeze(-1) * node_gate.unsqueeze(-2)

        hybrid_supports = []
        for support in supports:
            support = support.unsqueeze(0).expand(inputs.shape[0], -1, -1)
            hybrid_supports.append(self.graph_fusion * support + (1.0 - self.graph_fusion) * text_adaptive)
        hybrid_supports.append(text_adaptive)

        features = [inputs]
        for support in hybrid_supports:
            hidden = self._propagate(inputs, support)
            features.append(hidden)
            for _ in range(1, self.order):
                hidden = self._propagate(hidden, support)
                features.append(hidden)

        hidden = torch.cat(features, dim=-1)
        hidden = self.dropout(F.gelu(self.proj(hidden)))
        return self.norm(inputs + hidden)


class TextCrossformerEncoder(nn.Module):
    def __init__(self, seq_len, patch_len, hidden_dim, num_layers, num_heads, ff_dim, dropout):
        super(TextCrossformerEncoder, self).__init__()
        self.patch_len = patch_len
        self.output_tokens = (seq_len + patch_len - 1) // patch_len
        self.patch_proj = nn.Linear(patch_len * hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TextCrossformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def _patchify(self, inputs):
        batch_size, seq_len, node_num, hidden_dim = inputs.shape
        pad_len = self.output_tokens * self.patch_len - seq_len
        if pad_len > 0:
            inputs = torch.cat([inputs, inputs[:, -1:, :, :].expand(-1, pad_len, -1, -1)], dim=1)
        patches = inputs.view(batch_size, self.output_tokens, self.patch_len, node_num, hidden_dim)
        patches = patches.permute(0, 1, 3, 2, 4).reshape(batch_size, self.output_tokens, node_num, self.patch_len * hidden_dim)
        return self.patch_proj(patches)

    def forward(self, inputs, text_hidden):
        patch_tokens = self._patchify(inputs)
        text_tokens = self._patchify(text_hidden)
        hidden = patch_tokens
        for block in self.blocks:
            hidden = block(hidden, text_tokens)
        return hidden


class TextCrossformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout):
        super(TextCrossformerBlock, self).__init__()
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, patch_tokens, text_tokens):
        batch_size, patch_num, node_num, hidden_dim = patch_tokens.shape

        x = patch_tokens.permute(0, 2, 1, 3).reshape(batch_size * node_num, patch_num, hidden_dim)
        attn_out, _ = self.temporal_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        x = x.view(batch_size, node_num, patch_num, hidden_dim).permute(0, 2, 1, 3)

        gate = torch.sigmoid(self.text_gate(torch.cat([x, text_tokens], dim=-1)))
        x = self.norm2(x + gate * text_tokens)
        ff_out = self.ffn(x)
        return self.norm3(x + self.dropout(ff_out))


class RoadReportGenerator(nn.Module):
    def __init__(self, hidden_dim, report_len, vocab_size, memory_size, anomaly_topk, num_heads, dropout):
        super(RoadReportGenerator, self).__init__()
        self.anomaly_topk = anomaly_topk
        self.report_queries = nn.Parameter(torch.randn(report_len, hidden_dim))
        self.memory_bank = nn.Parameter(torch.randn(memory_size, hidden_dim))
        self.road_scorer = nn.Linear(hidden_dim, 1)
        self.road_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.memory_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, encoded_hidden):
        node_hidden = encoded_hidden.mean(dim=1)
        road_scores = torch.sigmoid(self.road_scorer(node_hidden)).squeeze(-1)

        topk = min(max(1, self.anomaly_topk), node_hidden.shape[1])
        top_indices = torch.topk(road_scores, k=topk, dim=-1).indices
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, node_hidden.shape[-1])
        anomaly_hidden = torch.gather(node_hidden, dim=1, index=gather_index)

        queries = self.report_queries.unsqueeze(0).expand(node_hidden.shape[0], -1, -1)
        road_context, _ = self.road_attn(queries, anomaly_hidden, anomaly_hidden, need_weights=False)
        memory = self.memory_bank.unsqueeze(0).expand(node_hidden.shape[0], -1, -1)
        memory_context, _ = self.memory_attn(queries + road_context, memory, memory, need_weights=False)
        hidden = self.norm(queries + road_context + memory_context)
        logits = self.output_proj(hidden)
        return logits, road_scores
