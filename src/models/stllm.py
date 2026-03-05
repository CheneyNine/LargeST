import warnings

import torch
import torch.nn as nn

from src.base.model import BaseModel

try:
    from transformers import GPT2Config
    from transformers.models.gpt2.modeling_gpt2 import GPT2Model
except Exception:
    GPT2Config = None
    GPT2Model = None


class TemporalEmbedding(nn.Module):
    def __init__(self, steps_per_day, features):
        super(TemporalEmbedding, self).__init__()
        self.steps_per_day = int(steps_per_day)

        self.time_day = nn.Parameter(torch.empty(self.steps_per_day, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def _extract_index(self, values, vocab_size):
        # Support normalized or integer timestamp encodings.
        max_val = values.detach().max()
        min_val = values.detach().min()
        if min_val >= 0 and max_val <= 1.5:
            index = torch.floor(values * vocab_size).long()
        else:
            index = values.long()
        index = torch.clamp(index, min=0, max=vocab_size - 1)
        return index

    def forward(self, x, time_day_idx, day_in_week_idx):
        # x: [B, T, N, F]
        batch_size, _, node_num, feat_dim = x.shape
        device = x.device
        emb_dim = self.time_day.shape[1]

        if time_day_idx < 0 or time_day_idx >= feat_dim:
            time_day = torch.zeros(batch_size, emb_dim, node_num, 1, device=device)
        else:
            day_feature = x[:, -1, :, time_day_idx]
            day_index = self._extract_index(day_feature, self.steps_per_day)
            day_emb = self.time_day[day_index]  # [B, N, D]
            time_day = day_emb.transpose(1, 2).unsqueeze(-1)

        if day_in_week_idx < 0 or day_in_week_idx >= feat_dim:
            time_week = torch.zeros(batch_size, emb_dim, node_num, 1, device=device)
        else:
            week_feature = x[:, -1, :, day_in_week_idx]
            week_index = self._extract_index(week_feature, 7)
            week_emb = self.time_week[week_index]  # [B, N, D]
            time_week = week_emb.transpose(1, 2).unsqueeze(-1)

        return time_day + time_week


class PFA(nn.Module):
    def __init__(
        self,
        gpt_model_name="gpt2",
        gpt_layers=6,
        U=1,
        local_files_only=True,
        allow_download=True,
    ):
        super(PFA, self).__init__()
        self.gpt2 = self._load_gpt2(
            gpt_model_name=gpt_model_name,
            local_files_only=local_files_only,
            allow_download=allow_download,
        )

        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.hidden_size = int(self.gpt2.config.hidden_size)
        self.U = max(0, int(U))
        self.gpt_layers = gpt_layers
        self._apply_partial_freeze()

    def _load_gpt2(self, gpt_model_name, local_files_only, allow_download):
        if GPT2Model is None:
            raise ImportError(
                "transformers is required by ST_LLM. Please install transformers first."
            )

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
                        "Failed to load pretrained GPT2 ({}, {}). Fall back to randomly "
                        "initialized GPT2Config.".format(first_error, second_error)
                    )
            else:
                warnings.warn(
                    "Failed local GPT2 load ({}). Fall back to randomly initialized GPT2Config.".format(
                        first_error
                    )
                )

            if GPT2Config is None:
                raise ImportError("transformers GPT2Config is unavailable.")
            return GPT2Model(GPT2Config())

    def _apply_partial_freeze(self):
        # Keep the partial frozen attention strategy used in ST-LLM.
        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                if layer_index < self.gpt_layers - self.U:
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

    def forward(self, x):
        # x: [B, N, hidden_size]
        return self.gpt2(inputs_embeds=x).last_hidden_state


class ST_LLM(BaseModel):
    """
    LargeST-compatible integration of ST-LLM's ST_LLM model.
    """

    def __init__(
        self,
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
        **args
    ):
        super(ST_LLM, self).__init__(**args)
        self.node_dim = channels
        self.llm_layer = llm_layer
        self.U = U
        self.steps_per_day = steps_per_day
        self.time_day_idx = time_day_idx
        self.day_in_week_idx = day_in_week_idx
        self.gpt_channel = gpt_channel

        self.temporal_embedding = TemporalEmbedding(self.steps_per_day, self.gpt_channel)
        self.node_emb = nn.Parameter(torch.empty(self.node_num, self.gpt_channel))
        nn.init.xavier_uniform_(self.node_emb)

        self.start_conv = nn.Conv2d(
            self.input_dim * self.seq_len,
            self.gpt_channel,
            kernel_size=(1, 1),
        )

        self.gpt = PFA(
            gpt_model_name=gpt_model_name,
            gpt_layers=self.llm_layer,
            U=self.U,
            local_files_only=bool(gpt_local_files_only),
            allow_download=bool(gpt_allow_download),
        )
        self.gpt_hidden = self.gpt.hidden_size

        self.feature_fusion = nn.Conv2d(
            self.gpt_channel * 3,
            self.gpt_hidden,
            kernel_size=(1, 1),
        )
        self.regression_layer = nn.Conv2d(
            self.gpt_hidden,
            self.horizon,
            kernel_size=(1, 1),
        )

    def forward(self, history_data, label=None):
        # history_data: [B, T, N, F]
        batch_size, seq_len, node_num, feat_dim = history_data.shape
        if node_num != self.node_num:
            raise ValueError(
                "Node num mismatch in ST_LLM forward: got {}, expected {}".format(
                    node_num, self.node_num
                )
            )
        if feat_dim != self.input_dim:
            raise ValueError(
                "Input dim mismatch in ST_LLM forward: got {}, expected {}".format(
                    feat_dim, self.input_dim
                )
            )
        if seq_len != self.seq_len:
            raise ValueError(
                "Seq len mismatch in ST_LLM forward: got {}, expected {}".format(
                    seq_len, self.seq_len
                )
            )

        temporal = self.temporal_embedding(
            history_data, self.time_day_idx, self.day_in_week_idx
        )  # [B, C, N, 1]
        node_emb = (
            self.node_emb.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .transpose(1, 2)
            .unsqueeze(-1)
        )  # [B, C, N, 1]

        # [B, T, N, F] -> [B, F, N, T] -> [B, F*T, N, 1]
        input_data = history_data.permute(0, 3, 2, 1).contiguous()
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, node_num, -1).transpose(1, 2).unsqueeze(-1)
        input_data = self.start_conv(input_data)  # [B, C, N, 1]

        data_st = torch.cat([input_data, temporal, node_emb], dim=1)  # [B, 3C, N, 1]
        data_st = self.feature_fusion(data_st)  # [B, gpt_hidden, N, 1]

        token = data_st.permute(0, 2, 1, 3).squeeze(-1)  # [B, N, gpt_hidden]
        token = self.gpt(token)  # [B, N, gpt_hidden]
        token = token.permute(0, 2, 1).unsqueeze(-1)  # [B, gpt_hidden, N, 1]

        prediction = self.regression_layer(token)  # [B, H, N, 1]
        return prediction
