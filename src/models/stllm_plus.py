import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base.model import BaseModel

try:
    from transformers import GPT2Config, GPT2Model
except Exception:
    GPT2Config = None
    GPT2Model = None

try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None


class TemporalEmbedding(nn.Module):
    def __init__(self, steps_per_day, features):
        super(TemporalEmbedding, self).__init__()
        self.steps_per_day = int(steps_per_day)
        self.time_day = nn.Parameter(torch.empty(self.steps_per_day, features))
        nn.init.xavier_uniform_(self.time_day)
        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def _extract_index(self, values, vocab_size):
        max_val = values.detach().max()
        min_val = values.detach().min()
        if min_val >= 0 and max_val <= 1.5:
            index = torch.floor(values * vocab_size).long()
        else:
            index = values.long()
        return torch.clamp(index, min=0, max=vocab_size - 1)

    def forward(self, x, time_day_idx, day_in_week_idx):
        batch_size, _, node_num, feat_dim = x.shape
        device = x.device
        emb_dim = self.time_day.shape[1]

        if time_day_idx < 0 or time_day_idx >= feat_dim:
            time_day = torch.zeros(batch_size, emb_dim, node_num, 1, device=device)
        else:
            day_feature = x[:, -1, :, time_day_idx]
            day_index = self._extract_index(day_feature, self.steps_per_day)
            day_emb = self.time_day[day_index]
            time_day = day_emb.transpose(1, 2).unsqueeze(-1)

        if day_in_week_idx < 0 or day_in_week_idx >= feat_dim:
            time_week = torch.zeros(batch_size, emb_dim, node_num, 1, device=device)
        else:
            week_feature = x[:, -1, :, day_in_week_idx]
            week_index = self._extract_index(week_feature, 7)
            week_emb = self.time_week[week_index]
            time_week = week_emb.transpose(1, 2).unsqueeze(-1)

        return time_day + time_week


class PartiallyFrozenGraphAttention(nn.Module):
    def __init__(
        self,
        gpt_model_name="gpt2",
        gpt_layers=6,
        U=1,
        dropout_rate=0.1,
        local_files_only=True,
        allow_download=True,
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1,
    ):
        super(PartiallyFrozenGraphAttention, self).__init__()
        self.gpt2 = self._load_gpt2(
            gpt_model_name=gpt_model_name,
            local_files_only=local_files_only,
            allow_download=allow_download,
        )
        self.gpt2.h = self.gpt2.h[: int(gpt_layers)]
        self.hidden_size = int(self.gpt2.config.hidden_size)
        self.n_head = int(self.gpt2.config.n_head)
        self.U = max(0, int(U))
        self.gpt_layers = int(gpt_layers)
        self.dropout = nn.Dropout(float(dropout_rate))

        if bool(use_lora):
            if LoraConfig is None or get_peft_model is None:
                raise ImportError(
                    "peft is required for STLLM+ LoRA training. Install peft or disable use_lora."
                )
            lora_config = LoraConfig(
                r=int(lora_rank),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=["c_attn"],
                bias="none",
            )
            self.gpt2 = get_peft_model(self.gpt2, lora_config)

        self._apply_partial_freeze()

    def _load_gpt2(self, gpt_model_name, local_files_only, allow_download):
        if GPT2Model is None:
            raise ImportError("transformers is required by STLLM+.")
        try:
            return GPT2Model.from_pretrained(
                gpt_model_name,
                output_attentions=True,
                output_hidden_states=True,
                local_files_only=bool(local_files_only),
            )
        except Exception as first_error:
            if allow_download:
                try:
                    return GPT2Model.from_pretrained(
                        gpt_model_name,
                        output_attentions=True,
                        output_hidden_states=True,
                        local_files_only=False,
                    )
                except Exception as second_error:
                    warnings.warn(
                        "Failed pretrained GPT2 load ({}, {}). Fall back to GPT2Config.".format(
                            first_error, second_error
                        )
                    )
            else:
                warnings.warn(
                    "Failed local GPT2 load ({}). Fall back to GPT2Config.".format(
                        first_error
                    )
                )
            if GPT2Config is None:
                raise ImportError("transformers GPT2Config is unavailable.")
            return GPT2Model(GPT2Config())

    def _apply_partial_freeze(self):
        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                if layer_index < self.gpt_layers - self.U:
                    if "ln" in name or "wpe" in name or "lora" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

    def _build_attention_mask(self, adjacency_matrix, batch_size, device, dtype):
        adjacency = adjacency_matrix.to(device=device)
        if adjacency.dim() != 2:
            raise ValueError("adjacency_matrix must be [N, N], got {}".format(tuple(adjacency.shape)))
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency_matrix must be square")

        eye = torch.eye(adjacency.shape[0], device=device, dtype=adjacency.dtype)
        allowed = (adjacency > 0) | (eye > 0)
        mask = torch.zeros_like(adjacency, dtype=dtype)
        mask = mask.masked_fill(~allowed, torch.finfo(dtype).min)
        mask = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, self.n_head, -1, -1)
        return mask

    def _forward_with_graph_mask(self, inputs_embeds, attention_mask):
        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

        hidden_states = inputs_embeds + self.gpt2.wpe(position_ids)
        all_hidden_states = []
        all_attentions = []

        for block in self.gpt2.h:
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
            )
            hidden_states = outputs[0]
            all_attentions.append(outputs[1])
            all_hidden_states.append(hidden_states)

        hidden_states = self.gpt2.ln_f(hidden_states)
        return hidden_states, tuple(all_hidden_states), tuple(all_attentions)

    def forward(self, x, adjacency_matrix):
        batch_size = x.shape[0]
        attention_mask = self._build_attention_mask(
            adjacency_matrix=adjacency_matrix,
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
        )
        hidden_states, _, _ = self._forward_with_graph_mask(x, attention_mask)
        return self.dropout(hidden_states)


class STLLMPlus(BaseModel):
    """
    LargeST-compatible ST-LLM+.

    Source alignment:
    - official temporal/node/start-conv pipeline
    - graph-masked GPT2 attention
    - LoRA-augmented partially frozen GPT2 blocks
    """

    def __init__(
        self,
        adj_mx,
        channels=64,
        llm_layer=6,
        U=1,
        steps_per_day=288,
        time_day_idx=1,
        day_in_week_idx=2,
        gpt_model_name="gpt2",
        gpt_local_files_only=1,
        gpt_allow_download=1,
        gpt_channel=256,
        use_lora=1,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1,
        dropout=0.1,
        **args
    ):
        super(STLLMPlus, self).__init__(**args)

        self.node_dim = int(channels)
        self.llm_layer = int(llm_layer)
        self.U = int(U)
        self.steps_per_day = int(steps_per_day)
        self.time_day_idx = int(time_day_idx)
        self.day_in_week_idx = int(day_in_week_idx)
        self.gpt_channel = int(gpt_channel)

        adjacency = torch.as_tensor(adj_mx, dtype=torch.float32)
        if adjacency.shape != (self.node_num, self.node_num):
            raise ValueError(
                "adj_mx shape mismatch: got {}, expected ({}, {})".format(
                    tuple(adjacency.shape), self.node_num, self.node_num
                )
            )
        self.register_buffer("adj_mx", adjacency, persistent=False)

        self.temporal_embedding = TemporalEmbedding(self.steps_per_day, self.gpt_channel)
        self.node_emb = nn.Parameter(torch.empty(self.node_num, self.gpt_channel))
        nn.init.xavier_uniform_(self.node_emb)

        self.start_conv = nn.Conv2d(
            self.input_dim * self.seq_len,
            self.gpt_channel,
            kernel_size=(1, 1),
        )

        self.gpt = PartiallyFrozenGraphAttention(
            gpt_model_name=gpt_model_name,
            gpt_layers=self.llm_layer,
            U=self.U,
            dropout_rate=dropout,
            local_files_only=bool(gpt_local_files_only),
            allow_download=bool(gpt_allow_download),
            use_lora=bool(use_lora),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        self.in_layer = nn.Conv2d(
            self.gpt_channel * 3,
            self.gpt.hidden_size,
            kernel_size=(1, 1),
        )
        self.regression_layer = nn.Conv2d(
            self.gpt.hidden_size,
            self.horizon,
            kernel_size=(1, 1),
        )

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, history_data, label=None):
        del label
        batch_size, seq_len, node_num, feat_dim = history_data.shape
        if seq_len != self.seq_len:
            raise ValueError(
                "Seq len mismatch in STLLMPlus: got {}, expected {}".format(
                    seq_len, self.seq_len
                )
            )
        if node_num != self.node_num:
            raise ValueError(
                "Node num mismatch in STLLMPlus: got {}, expected {}".format(
                    node_num, self.node_num
                )
            )
        if feat_dim != self.input_dim:
            raise ValueError(
                "Input dim mismatch in STLLMPlus: got {}, expected {}".format(
                    feat_dim, self.input_dim
                )
            )

        temporal = self.temporal_embedding(
            history_data, self.time_day_idx, self.day_in_week_idx
        )
        node_emb = (
            self.node_emb.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .transpose(1, 2)
            .unsqueeze(-1)
        )

        input_data = history_data.permute(0, 3, 2, 1).contiguous()
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, node_num, -1).transpose(1, 2).unsqueeze(-1)
        input_data = self.start_conv(input_data)

        data_st = torch.cat([input_data, temporal, node_emb], dim=1)
        data_st = F.leaky_relu(self.in_layer(data_st))
        data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)

        outputs = self.gpt(data_st, self.adj_mx)
        outputs = outputs.permute(0, 2, 1).unsqueeze(-1)
        outputs = self.regression_layer(outputs)
        return outputs
