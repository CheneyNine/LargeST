import os
import warnings

import torch
import torch.nn as nn

from src.base.model import BaseModel

try:
    from transformers import GPT2Config, GPT2Model
except Exception:
    GPT2Config = None
    GPT2Model = None

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception:
    LoraConfig = None
    TaskType = None
    get_peft_model = None


def ensure_word_embedding_pca(
    cache_path,
    model_name="gpt2",
    local_files_only=True,
    allow_download=True,
    n_components=500,
):
    if cache_path and os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location="cpu")
        return torch.as_tensor(cached, dtype=torch.float32)

    if GPT2Model is None:
        raise ImportError(
            "transformers is required to generate CALF PCA word embeddings."
        )

    try:
        model = GPT2Model.from_pretrained(
            model_name,
            output_attentions=False,
            output_hidden_states=False,
            local_files_only=bool(local_files_only),
        )
    except Exception as first_error:
        if not allow_download:
            raise
        try:
            model = GPT2Model.from_pretrained(
                model_name,
                output_attentions=False,
                output_hidden_states=False,
                local_files_only=False,
            )
        except Exception as second_error:
            raise RuntimeError(
                "Failed to load GPT2 for CALF PCA cache generation: {}; {}".format(
                    first_error, second_error
                )
            )

    with torch.no_grad():
        wte = model.wte.weight.detach().cpu().float().transpose(0, 1).contiguous()
        centered = wte - wte.mean(dim=0, keepdim=True)
        q = min(int(n_components), centered.shape[0], centered.shape[1])
        U, S, _ = torch.pca_lowrank(centered, q=q)
        transformed = U[:, :q] * S[:q]

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save(transformed.cpu(), cache_path)
    return transformed.cpu()


class EncoderPCA(nn.Module):
    def __init__(
        self,
        input_dim,
        word_embedding,
        hidden_dim=768,
        num_heads=4,
        num_encoder_layers=1,
    ):
        super(EncoderPCA, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.register_buffer(
            "word_embedding",
            word_embedding.transpose(0, 1).contiguous().float(),
            persistent=False,
        )

    def forward(self, x):
        # x: [B, N, L]
        batch_size = x.shape[0]
        x = self.linear(x)
        x = self.transformer_encoder(x)
        x_time = x

        word_embedding = self.word_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        x_text, _ = self.cross_attention(x_time, word_embedding, word_embedding)
        return x_time, x_text


class CALF(BaseModel):
    """
    LargeST-compatible long-term forecasting adaptation of CALF.

    This integration follows the official architecture:
    Encoder_PCA + dual GPT2 branches + feature/output alignment.
    """

    def __init__(
        self,
        traffic_dim,
        d_model,
        n_heads,
        gpt_layers,
        pca_encoder_layers,
        pretrain,
        use_lora,
        lora_rank,
        lora_alpha,
        lora_dropout,
        gpt_model_name,
        gpt_local_files_only,
        gpt_allow_download,
        word_embedding_path,
        word_embedding_components,
        **args
    ):
        super(CALF, self).__init__(**args)

        self.traffic_dim = int(traffic_dim)
        if self.traffic_dim != 1:
            raise ValueError(
                "CALF currently supports flow-only forecasting in LargeST. "
                "Set traffic_dim=1, got {}".format(self.traffic_dim)
            )
        if self.traffic_dim > self.input_dim:
            raise ValueError(
                "traffic_dim must be <= input_dim, got {} > {}".format(
                    self.traffic_dim, self.input_dim
                )
            )

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.gpt_layers = int(gpt_layers)
        self.pretrain = bool(pretrain)
        self.use_lora = bool(use_lora)
        self.gpt_model_name = str(gpt_model_name).strip() or "gpt2"
        self.gpt_local_files_only = bool(gpt_local_files_only)
        self.gpt_allow_download = bool(gpt_allow_download)

        word_embedding = ensure_word_embedding_pca(
            cache_path=word_embedding_path,
            model_name=self.gpt_model_name,
            local_files_only=self.gpt_local_files_only,
            allow_download=self.gpt_allow_download,
            n_components=word_embedding_components,
        )

        self.time_gpt = self._load_gpt2()
        self.text_gpt = self._load_gpt2()
        self.time_gpt.h = self.time_gpt.h[: self.gpt_layers]
        self.text_gpt.h = self.text_gpt.h[: self.gpt_layers]

        if self.use_lora:
            if LoraConfig is None or TaskType is None or get_peft_model is None:
                raise ImportError("peft is required for CALF LoRA training.")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=int(lora_rank),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=["c_attn"],
            )
            self.time_gpt = get_peft_model(self.time_gpt, peft_config)

        self._freeze_parameters()

        self.time_proj = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_model, bias=False) for _ in range(self.gpt_layers + 1)]
        )
        self.text_proj = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_model, bias=False) for _ in range(self.gpt_layers + 1)]
        )

        self.in_layer = EncoderPCA(
            input_dim=self.seq_len,
            word_embedding=word_embedding,
            hidden_dim=self.d_model,
            num_heads=self.n_heads,
            num_encoder_layers=int(pca_encoder_layers),
        )
        self.out_layer = nn.Linear(self.d_model, self.horizon)

    def _load_gpt2(self):
        if GPT2Model is None:
            raise ImportError("transformers is required by CALF.")
        if self.pretrain:
            try:
                return GPT2Model.from_pretrained(
                    self.gpt_model_name,
                    output_attentions=True,
                    output_hidden_states=True,
                    local_files_only=self.gpt_local_files_only,
                )
            except Exception as first_error:
                if not self.gpt_allow_download:
                    raise
                try:
                    return GPT2Model.from_pretrained(
                        self.gpt_model_name,
                        output_attentions=True,
                        output_hidden_states=True,
                        local_files_only=False,
                    )
                except Exception as second_error:
                    raise RuntimeError(
                        "Failed to load GPT2 for CALF: {}; {}".format(
                            first_error, second_error
                        )
                    )

        if GPT2Config is None:
            raise ImportError("transformers GPT2Config is unavailable.")
        warnings.warn("CALF is using randomly initialized GPT2Config.")
        return GPT2Model(GPT2Config())

    def _freeze_parameters(self):
        for name, param in self.time_gpt.named_parameters():
            if "ln" in name or "wpe" in name or "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for name, param in self.text_gpt.named_parameters():
            if "wpe" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, history_data, label=None):
        del label
        x = history_data[..., : self.traffic_dim]
        batch_size, seq_len, node_num, feat_dim = x.shape
        if feat_dim != 1:
            raise ValueError("CALF expects one traffic feature, got {}".format(feat_dim))
        if seq_len != self.seq_len:
            raise ValueError(
                "Seq len mismatch in CALF: got {}, expected {}".format(
                    seq_len, self.seq_len
                )
            )
        if node_num != self.node_num:
            raise ValueError(
                "Node num mismatch in CALF: got {}, expected {}".format(
                    node_num, self.node_num
                )
            )

        x = x.squeeze(-1)  # [B, T, N]
        means = x.mean(dim=1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / stdev

        x = x.permute(0, 2, 1).contiguous()  # [B, N, T]
        outputs_time0, outputs_text0 = self.in_layer(x)

        outputs_time_raw = self.time_gpt(
            inputs_embeds=outputs_time0,
            output_hidden_states=True,
            return_dict=True,
        )
        outputs_text_raw = self.text_gpt(
            inputs_embeds=outputs_text0,
            output_hidden_states=True,
            return_dict=True,
        )

        outputs_time = outputs_time_raw.last_hidden_state + outputs_time0
        outputs_text = outputs_text_raw.last_hidden_state + outputs_text0

        intermediate_time = tuple(
            self.time_proj[idx](feat)
            for idx, feat in enumerate(outputs_time_raw.hidden_states[: len(self.time_proj)])
        )
        intermediate_text = tuple(
            self.text_proj[idx](feat)
            for idx, feat in enumerate(outputs_text_raw.hidden_states[: len(self.text_proj)])
        )

        outputs_time = self.out_layer(outputs_time[:, -node_num:, :]).permute(0, 2, 1).contiguous()
        outputs_text = self.out_layer(outputs_text[:, -node_num:, :]).permute(0, 2, 1).contiguous()

        outputs_time = outputs_time * stdev + means
        outputs_text = outputs_text * stdev + means

        return {
            "outputs_time": outputs_time.unsqueeze(-1),
            "outputs_text": outputs_text.unsqueeze(-1),
            "intermidiate_time": intermediate_time,
            "intermidiate_text": intermediate_text,
        }
