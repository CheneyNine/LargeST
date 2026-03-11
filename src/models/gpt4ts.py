import warnings

import torch
import torch.nn as nn

from src.base.model import BaseModel

try:
    from transformers import GPT2Config, GPT2Model
except Exception:
    GPT2Config = None
    GPT2Model = None


class GPT4TS(BaseModel):
    """
    LargeST adaptation of the official GPT4TS long-term forecasting model.

    The original model treats each variable as one scalar time series. In the
    traffic setting here, the supported and faithful path is flow-only
    forecasting, i.e. the first `traffic_dim` channel should normally be 1.
    """

    def __init__(
        self,
        traffic_dim,
        d_model,
        patch_size,
        stride,
        gpt_layers,
        is_gpt,
        pretrain,
        freeze,
        dropout,
        gpt_model_name="gpt2",
        gpt_local_files_only=1,
        gpt_allow_download=1,
        **args
    ):
        super(GPT4TS, self).__init__(**args)

        self.traffic_dim = int(traffic_dim)
        if self.traffic_dim != 1:
            raise ValueError(
                "GPT4TS currently supports flow-only forecasting in LargeST. "
                "Set traffic_dim=1, got {}".format(self.traffic_dim)
            )
        if self.traffic_dim > self.input_dim:
            raise ValueError(
                "traffic_dim must be <= input_dim, got {} > {}".format(
                    self.traffic_dim, self.input_dim
                )
            )
        if patch_size <= 0 or stride <= 0:
            raise ValueError("patch_size and stride must be > 0")
        if self.seq_len + stride < patch_size:
            raise ValueError(
                "seq_len + stride must be >= patch_size, got {} + {} < {}".format(
                    self.seq_len, stride, patch_size
                )
            )

        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.d_model = int(d_model)
        self.gpt_layers = int(gpt_layers)
        self.is_gpt = bool(is_gpt)
        self.pretrain = bool(pretrain)
        self.freeze = bool(freeze)
        self.dropout = nn.Dropout(float(dropout))
        self.gpt_model_name = str(gpt_model_name).strip() or "gpt2"
        self.gpt_local_files_only = bool(gpt_local_files_only)
        self.gpt_allow_download = bool(gpt_allow_download)

        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        self.gpt2 = None
        if self.is_gpt:
            self.gpt2 = self._load_gpt2()
            self.gpt2.h = self.gpt2.h[: self.gpt_layers]

        self.in_layer = nn.Linear(self.patch_size, self.d_model)
        self.out_layer = nn.Linear(self.d_model * self.patch_num, self.horizon)

    def _load_gpt2(self):
        if GPT2Model is None:
            raise ImportError(
                "transformers is required for GPT4TS. Please install transformers."
            )

        if self.pretrain:
            try:
                model = GPT2Model.from_pretrained(
                    self.gpt_model_name,
                    output_attentions=True,
                    output_hidden_states=True,
                    local_files_only=self.gpt_local_files_only,
                )
            except Exception as first_error:
                if not self.gpt_allow_download:
                    raise
                try:
                    model = GPT2Model.from_pretrained(
                        self.gpt_model_name,
                        output_attentions=True,
                        output_hidden_states=True,
                        local_files_only=False,
                    )
                except Exception as second_error:
                    raise RuntimeError(
                        "Failed to load pretrained GPT2 for GPT4TS: {}; {}".format(
                            first_error, second_error
                        )
                    )
        else:
            if GPT2Config is None:
                raise ImportError("transformers GPT2Config is unavailable.")
            warnings.warn("GPT4TS is using randomly initialized GPT2Config.")
            model = GPT2Model(GPT2Config())

        if self.freeze and self.pretrain:
            for name, param in model.named_parameters():
                if "ln" in name or "wpe" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        return model

    def forward(self, history_data, label=None):
        del label
        x = history_data[..., : self.traffic_dim]
        batch_size, seq_len, node_num, feat_dim = x.shape
        if feat_dim != 1:
            raise ValueError("GPT4TS expects one traffic feature, got {}".format(feat_dim))
        if seq_len != self.seq_len:
            raise ValueError(
                "Seq len mismatch in GPT4TS forward: got {}, expected {}".format(
                    seq_len, self.seq_len
                )
            )
        if node_num != self.node_num:
            raise ValueError(
                "Node num mismatch in GPT4TS forward: got {}, expected {}".format(
                    node_num, self.node_num
                )
            )

        x = x.squeeze(-1)  # [B, T, N]
        means = x.mean(dim=1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / stdev

        x = x.permute(0, 2, 1).contiguous()  # [B, N, T]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = x.contiguous().view(batch_size * node_num, self.patch_num, self.patch_size)

        outputs = self.in_layer(x)
        if self.gpt2 is not None:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state
        outputs = self.dropout(outputs)

        outputs = self.out_layer(outputs.reshape(batch_size * node_num, -1))
        outputs = outputs.view(batch_size, node_num, self.horizon).permute(0, 2, 1).contiguous()
        outputs = outputs * stdev + means
        return outputs.unsqueeze(-1)
