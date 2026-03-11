import math
import warnings

import torch
import torch.nn as nn

from src.base.model import BaseModel

try:
    from transformers import BertConfig, BertModel, BertTokenizer
    from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
    from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
except Exception:
    BertConfig = None
    BertModel = None
    BertTokenizer = None
    GPT2Config = None
    GPT2Model = None
    GPT2Tokenizer = None
    LlamaConfig = None
    LlamaModel = None
    LlamaTokenizer = None


DEFAULT_LLM_MODEL_NAME = {
    "GPT2": "openai-community/gpt2",
    "BERT": "google-bert/bert-base-uncased",
    "LLAMA": "huggyllama/llama-7b",
}

DESCRIPTION_BANK = {
    "ETTH1": "Electricity transformer temperature time series.",
    "ETTH2": "Electricity transformer temperature time series.",
    "ETTM1": "Electricity transformer temperature time series.",
    "ETTM2": "Electricity transformer temperature time series.",
    "ECL": "Electricity consumption time series.",
    "WEATHER": "Weather observations time series.",
    "TRAFFIC": "Road traffic occupancy time series.",
    "CA": "California freeway traffic time series.",
    "GLA": "Greater Los Angeles traffic time series.",
    "GBA": "Greater Bay Area traffic time series.",
    "SD": "San Diego traffic flow time series.",
    "SACRA": "Sacramento traffic flow time series.",
    "SACRAMENTO": "Sacramento traffic flow time series.",
}


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0.0):
        super(FlattenHead, self).__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


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
        attention = self.dropout(torch.softmax(scores * scale, dim=-1))
        reprogrammed = torch.einsum("bhls,she->blhe", attention, value)
        reprogrammed = reprogrammed.reshape(batch_size, seq_len, -1)
        return self.out_projection(reprogrammed)


class TimeLLM(BaseModel):
    """
    LargeST integration of Time-LLM components.

    Supported prompt modes:
    - stats: efficient numeric prompt tokens (recommended for large node_num)
    - text: official-style text prompt + tokenizer embeddings
    """

    def __init__(
        self,
        traffic_dim,
        d_model,
        llm_dim,
        llm_layers,
        llm_model,
        llm_model_name,
        llm_local_files_only,
        llm_allow_download,
        llm_torch_dtype,
        d_ff,
        n_heads,
        patch_len,
        stride,
        prompt_len,
        prompt_mode,
        prompt_granularity,
        prompt_domain,
        prompt_text,
        prompt_max_tokens,
        prompt_batch_size,
        node_chunk_size,
        num_prototypes,
        top_k_lags,
        freeze_backbone,
        use_revin,
        dropout,
        dataset_name="SD",
        **args
    ):
        super(TimeLLM, self).__init__(**args)

        if self.output_dim != 1:
            raise ValueError(
                "TimeLLM expects output_dim=1 in LargeST, got {}".format(self.output_dim)
            )
        if traffic_dim <= 0 or traffic_dim > self.input_dim:
            raise ValueError(
                "traffic_dim must be in (0, input_dim], got {} with input_dim={}".format(
                    traffic_dim, self.input_dim
                )
            )
        if patch_len <= 0 or stride <= 0:
            raise ValueError("patch_len and stride must be > 0")
        if prompt_len <= 0:
            raise ValueError("prompt_len must be > 0")
        if num_prototypes <= 0:
            raise ValueError("num_prototypes must be > 0")

        prompt_mode = str(prompt_mode).lower().strip()
        prompt_granularity = str(prompt_granularity).lower().strip()
        if prompt_mode not in {"stats", "text"}:
            raise ValueError("prompt_mode must be one of {'stats', 'text'}")
        if prompt_granularity not in {"batch", "node"}:
            raise ValueError("prompt_granularity must be one of {'batch', 'node'}")

        self.traffic_dim = int(traffic_dim)
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.n_heads = int(n_heads)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.prompt_len = int(prompt_len)
        self.prompt_mode = prompt_mode
        self.prompt_granularity = prompt_granularity
        self.prompt_max_tokens = int(prompt_max_tokens)
        self.prompt_batch_size = max(1, int(prompt_batch_size))
        self.node_chunk_size = max(0, int(node_chunk_size))
        self.top_k_lags = max(1, int(top_k_lags))
        self.freeze_backbone = bool(freeze_backbone)
        self.use_revin = bool(use_revin)
        self.checkpoint_trainable_only = self.freeze_backbone

        self.dataset_name = str(dataset_name)
        self.prompt_domain = bool(prompt_domain)
        self.prompt_text = str(prompt_text).strip()
        self.dataset_description = self._resolve_description()
        self.task_description = (
            "Forecast the next {} steps given the previous {} steps.".format(
                self.horizon, self.seq_len
            )
        )

        self.llm_model_type = str(llm_model).upper().strip()
        self.llm_model_name = str(llm_model_name).strip()
        self.llm_local_files_only = bool(llm_local_files_only)
        self.llm_allow_download = bool(llm_allow_download)
        self.llm_torch_dtype = self._resolve_torch_dtype(llm_torch_dtype)
        self._warned_text_fallback = False

        self.llm_model, self.tokenizer, loaded_llm_dim = self._load_llm_backbone(llm_layers)
        self.d_llm = int(loaded_llm_dim)
        if int(llm_dim) != self.d_llm:
            warnings.warn(
                "llm_dim={} differs from loaded backbone hidden_size={}. "
                "Use hidden_size={} automatically.".format(llm_dim, self.d_llm, self.d_llm)
            )

        if self.d_llm % self.n_heads != 0:
            raise ValueError("loaded llm hidden size must be divisible by n_heads")
        if self.seq_len + self.stride < self.patch_len:
            raise ValueError(
                "seq_len + stride must be >= patch_len, got {} + {} < {}".format(
                    self.seq_len, self.stride, self.patch_len
                )
            )

        self.patch_nums = int(math.floor((self.seq_len + self.stride - self.patch_len) / self.stride) + 1)
        if self.patch_nums <= 0:
            raise ValueError("Invalid patch settings produce non-positive patch_nums")

        self.patch_proj = nn.Linear(self.patch_len * self.traffic_dim, self.d_model)
        self.patch_norm = nn.LayerNorm(self.d_model)
        self.patch_dropout = nn.Dropout(dropout)

        self.prompt_stats_proj = nn.Sequential(
            nn.Linear(8, self.d_llm),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_llm, self.d_llm),
        )
        self.prompt_base = nn.Parameter(torch.randn(1, self.prompt_len, self.d_llm) * 0.02)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = int(self.word_embeddings.shape[0])
        self.num_prototypes = int(num_prototypes)
        # Keep trainable mapping weights in fp32 even when the frozen LLM backbone
        # runs in fp16/bf16. Pure fp16 Adam updates on this layer were unstable.
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_prototypes)
        self.reprogramming_layer = ReprogrammingLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_keys=self.d_model // self.n_heads,
            d_llm=self.d_llm,
            attention_dropout=dropout,
        )

        self.hidden_to_ff = nn.Linear(self.d_llm, self.d_ff)
        self.output_projection = FlattenHead(
            n_vars=self.node_num,
            nf=self.d_ff * self.patch_nums,
            target_window=self.horizon,
            head_dropout=dropout,
        )

    def _resolve_description(self):
        if self.prompt_domain and self.prompt_text:
            return self.prompt_text
        key = self.dataset_name.upper()
        return DESCRIPTION_BANK.get(key, "General multivariate traffic time series.")

    def _freeze_backbone(self, backbone):
        if not self.freeze_backbone:
            return
        for parameter in backbone.parameters():
            parameter.requires_grad = False

    def _select_llm_classes(self):
        if self.llm_model_type == "GPT2":
            return GPT2Config, GPT2Model, GPT2Tokenizer, DEFAULT_LLM_MODEL_NAME["GPT2"]
        if self.llm_model_type == "BERT":
            return BertConfig, BertModel, BertTokenizer, DEFAULT_LLM_MODEL_NAME["BERT"]
        if self.llm_model_type == "LLAMA":
            return LlamaConfig, LlamaModel, LlamaTokenizer, DEFAULT_LLM_MODEL_NAME["LLAMA"]
        raise ValueError("Unsupported llm_model: {}".format(self.llm_model_type))

    def _resolve_torch_dtype(self, dtype_name):
        if dtype_name is None:
            return torch.float32
        norm = str(dtype_name).strip().lower()
        if norm in {"float32", "fp32"}:
            return torch.float32
        if norm in {"float16", "fp16", "half"}:
            return torch.float16
        if norm in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if norm in {"auto", ""}:
            return None
        warnings.warn(
            "Unknown llm_torch_dtype={}, fallback to float32".format(dtype_name)
        )
        return torch.float32

    def _load_tokenizer(self, tokenizer_cls, model_name):
        tokenizer = None
        try:
            tokenizer = tokenizer_cls.from_pretrained(
                model_name,
                local_files_only=self.llm_local_files_only,
            )
        except Exception:
            if self.llm_allow_download:
                try:
                    tokenizer = tokenizer_cls.from_pretrained(
                        model_name,
                        local_files_only=False,
                    )
                except Exception as error:
                    warnings.warn("Tokenizer load failed for {}: {}".format(model_name, error))
            else:
                warnings.warn("Tokenizer local load failed for {}.".format(model_name))

        if tokenizer is not None and tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        return tokenizer

    def _load_llm_backbone(self, llm_layers):
        config_cls, model_cls, tokenizer_cls, default_name = self._select_llm_classes()
        if config_cls is None or model_cls is None:
            raise ImportError("transformers is required for TimeLLM backbone loading")

        model_name = self.llm_model_name if self.llm_model_name else default_name

        config = None
        try:
            config = config_cls.from_pretrained(
                model_name,
                local_files_only=self.llm_local_files_only,
            )
        except Exception:
            if self.llm_allow_download:
                try:
                    config = config_cls.from_pretrained(model_name, local_files_only=False)
                except Exception:
                    config = config_cls()
            else:
                config = config_cls()

        if hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = int(llm_layers)
        if hasattr(config, "output_attentions"):
            config.output_attentions = False
        if hasattr(config, "output_hidden_states"):
            config.output_hidden_states = False
        if hasattr(config, "use_cache"):
            config.use_cache = False

        model = None
        model_load_kwargs = {
            "config": config,
            "local_files_only": self.llm_local_files_only,
            "trust_remote_code": True,
        }
        if self.llm_torch_dtype is not None:
            model_load_kwargs["torch_dtype"] = self.llm_torch_dtype
        try:
            model = model_cls.from_pretrained(model_name, **model_load_kwargs)
        except Exception:
            if self.llm_allow_download:
                try:
                    model_load_kwargs["local_files_only"] = False
                    model = model_cls.from_pretrained(model_name, **model_load_kwargs)
                except Exception:
                    model = model_cls(config)
            else:
                model = model_cls(config)
        if self.llm_torch_dtype is not None:
            model = model.to(dtype=self.llm_torch_dtype)
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass

        tokenizer = None
        if tokenizer_cls is not None and self.prompt_mode == "text":
            tokenizer = self._load_tokenizer(tokenizer_cls, model_name)
            if tokenizer is not None:
                try:
                    model.resize_token_embeddings(len(tokenizer))
                except Exception:
                    pass

        self.llm_model_name = model_name
        self._freeze_backbone(model)
        hidden_size = int(model.config.hidden_size)
        return model, tokenizer, hidden_size

    def _normalize(self, inputs):
        # inputs: [B, T, N, C]
        if not self.use_revin:
            batch_size, _, node_num, feat_dim = inputs.shape
            mean = inputs.new_zeros(batch_size, 1, node_num, feat_dim)
            std = inputs.new_ones(batch_size, 1, node_num, feat_dim)
            return inputs, mean, std
        mean = inputs.mean(dim=1, keepdim=True)
        std = inputs.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        return (inputs - mean) / std, mean, std

    def _patchify(self, inputs):
        # inputs: [B, T, N, C]
        batch_size, _, node_num, _ = inputs.shape
        node_series = inputs.permute(0, 2, 1, 3).contiguous()  # [B, N, T, C]
        if self.stride > 0:
            pad = node_series[:, :, -1:, :].expand(-1, -1, self.stride, -1)
            node_series = torch.cat([node_series, pad], dim=2)

        # [B, N, P, patch_len, C]
        patches = node_series.unfold(dimension=2, size=self.patch_len, step=self.stride)
        patches = patches.contiguous().view(
            batch_size * node_num, self.patch_nums, self.patch_len * self.traffic_dim
        )

        patch_tokens = self.patch_proj(patches)
        patch_tokens = self.patch_norm(patch_tokens)
        patch_tokens = self.patch_dropout(patch_tokens)
        return patch_tokens

    def _calc_lags(self, series):
        # series: [S, T]
        if series.shape[-1] <= 1:
            return torch.zeros(series.shape[0], 1, device=series.device, dtype=torch.long)
        fft = torch.fft.rfft(series, dim=-1)
        corr = torch.fft.irfft(fft * torch.conj(fft), dim=-1)
        corr[:, 0] = float("-inf")
        lag_k = min(self.top_k_lags, corr.shape[-1] - 1)
        _, lags = torch.topk(corr, k=lag_k, dim=-1)
        return lags

    def _build_stats(self, series):
        # series: [S, T]
        min_v = series.min(dim=-1).values
        max_v = series.max(dim=-1).values
        median_v = series.median(dim=-1).values
        mean_v = series.mean(dim=-1)
        std_v = series.std(dim=-1, unbiased=False)
        trend_v = series[:, -1] - series[:, 0]

        centered = series - series.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(centered, dim=-1).abs()
        if spectrum.shape[-1] <= 1:
            period = torch.ones_like(min_v)
            strength = torch.zeros_like(min_v)
        else:
            spectrum = spectrum[:, 1:]
            peak_idx = spectrum.argmax(dim=-1) + 1
            period = float(series.shape[-1]) / peak_idx.float()
            strength = spectrum.max(dim=-1).values / (spectrum.mean(dim=-1) + 1e-6)

        lags = self._calc_lags(series).float().mean(dim=-1)
        return torch.stack(
            [min_v, max_v, median_v, mean_v, std_v, trend_v, period, strength + lags], dim=-1
        )

    def _stats_prompt_embeddings(self, series):
        # series: [S, T]
        stats = self._build_stats(series)
        stats_embed = self.prompt_stats_proj(stats).unsqueeze(1)  # [S, 1, D]
        return self.prompt_base.expand(series.shape[0], -1, -1) + stats_embed

    def _build_text_prompts(self, series, lags):
        # series: [S, T], lags: [S, K]
        stats = self._build_stats(series)
        prompts = []
        lag_count = lags.shape[-1]
        for idx in range(series.shape[0]):
            trend_word = "upward" if stats[idx, 5].item() > 0 else "downward"
            lag_values = ", ".join([str(int(v)) for v in lags[idx].tolist()])
            prompt = (
                "Dataset: {} Task: {} "
                "Input stats: min {:.4f}, max {:.4f}, median {:.4f}, mean {:.4f}, std {:.4f}, "
                "trend {}, top {} lags [{}]."
            ).format(
                self.dataset_description,
                self.task_description,
                stats[idx, 0].item(),
                stats[idx, 1].item(),
                stats[idx, 2].item(),
                stats[idx, 3].item(),
                stats[idx, 4].item(),
                trend_word,
                lag_count,
                lag_values,
            )
            prompts.append(prompt)
        return prompts

    def _text_prompt_embeddings(self, series):
        # series: [S, T]
        if self.tokenizer is None:
            if not self._warned_text_fallback:
                warnings.warn(
                    "Tokenizer is unavailable for text prompt mode, fallback to stats prompt mode."
                )
                self._warned_text_fallback = True
            return self._stats_prompt_embeddings(series)

        lags = self._calc_lags(series)
        prompts = self._build_text_prompts(series, lags)
        embedding_layer = self.llm_model.get_input_embeddings()

        all_embeddings = []
        for start in range(0, len(prompts), self.prompt_batch_size):
            end = min(len(prompts), start + self.prompt_batch_size)
            tokenized = self.tokenizer(
                prompts[start:end],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.prompt_max_tokens,
            )
            tokenized = {k: v.to(series.device) for k, v in tokenized.items()}
            prompt_emb = embedding_layer(tokenized["input_ids"])
            all_embeddings.append(prompt_emb)

        return torch.cat(all_embeddings, dim=0)

    def _prompt_embeddings(self, norm_inputs):
        # norm_inputs: [B, T, N, C]
        batch_size, _, node_num, _ = norm_inputs.shape
        base_series = norm_inputs[..., 0]  # [B, T, N]

        if self.prompt_granularity == "batch":
            prompt_series = base_series.mean(dim=2)  # [B, T]
            if self.prompt_mode == "text":
                prompt_emb = self._text_prompt_embeddings(prompt_series)
            else:
                prompt_emb = self._stats_prompt_embeddings(prompt_series)
            return prompt_emb.repeat_interleave(node_num, dim=0)

        # node-level prompts
        prompt_series = base_series.permute(0, 2, 1).contiguous().view(batch_size * node_num, self.seq_len)
        if self.prompt_mode == "text":
            return self._text_prompt_embeddings(prompt_series)
        return self._stats_prompt_embeddings(prompt_series)

    def forward(self, inputs, label=None):
        # inputs: [B, T, N, F]
        traffic_inputs = inputs[..., : self.traffic_dim]
        batch_size, _, node_num, _ = traffic_inputs.shape
        if node_num != self.node_num:
            raise ValueError(
                "Node num mismatch in TimeLLM forward: got {}, expected {}".format(
                    node_num, self.node_num
                )
            )

        norm_inputs, mean, std = self._normalize(traffic_inputs)
        prompt_embeddings = self._prompt_embeddings(norm_inputs)
        patch_tokens = self._patchify(norm_inputs)

        word_embeddings = self.word_embeddings
        mapping_dtype = self.mapping_layer.weight.dtype
        if word_embeddings.dtype != mapping_dtype:
            word_embeddings = word_embeddings.to(mapping_dtype)
        source_embeddings = self.mapping_layer(word_embeddings.permute(1, 0)).permute(1, 0)
        if source_embeddings.dtype != patch_tokens.dtype:
            source_embeddings = source_embeddings.to(patch_tokens.dtype)
        reprog_tokens = self.reprogramming_layer(
            patch_tokens,
            source_embeddings,
            source_embeddings,
        )

        sample_num = reprog_tokens.shape[0]
        chunk_size = self.node_chunk_size if self.node_chunk_size > 0 else sample_num
        llm_hidden_chunks = []
        for start in range(0, sample_num, chunk_size):
            end = min(sample_num, start + chunk_size)
            llm_inputs = torch.cat([prompt_embeddings[start:end], reprog_tokens[start:end]], dim=1)
            llm_inputs = llm_inputs.to(self.word_embeddings.dtype)
            llm_hidden = self.llm_model(
                inputs_embeds=llm_inputs,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
            ).last_hidden_state
            if llm_hidden.dtype != self.hidden_to_ff.weight.dtype:
                llm_hidden = llm_hidden.to(self.hidden_to_ff.weight.dtype)
            llm_hidden = self.hidden_to_ff(llm_hidden)
            llm_hidden = llm_hidden[:, -self.patch_nums :, :]  # keep patch tokens
            llm_hidden_chunks.append(llm_hidden)
        llm_hidden = torch.cat(llm_hidden_chunks, dim=0)

        llm_hidden = llm_hidden.view(batch_size, node_num, self.patch_nums, self.d_ff)
        llm_hidden = llm_hidden.permute(0, 1, 3, 2).contiguous()  # [B, N, d_ff, P]

        prediction = self.output_projection(llm_hidden)  # [B, N, H]
        prediction = prediction.permute(0, 2, 1).unsqueeze(-1).contiguous()  # [B, H, N, 1]

        if self.use_revin:
            target_mean = mean[..., : self.output_dim]
            target_std = std[..., : self.output_dim]
            prediction = prediction * target_std + target_mean

        return prediction
