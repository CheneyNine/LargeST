import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base.model import BaseModel

try:
    from transformers import GPT2Tokenizer, GPT2Model
except Exception:
    GPT2Model = None
    GPT2Tokenizer = None


OFFICIAL_INPUT_TEMPLATES = {
    "FRED": "From [t1] to [t2], the values were value1, ..., valuen every month. The total trend value was Trends",
    "ILI": "From [t1] to [t2], the values were value1, ..., valuen every week. The total trend value was Trends",
    "ETTh1": "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
    "ETTh2": "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
    "ECL": "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
    "ETTm1": "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
    "ETTm2": "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
    "Weather": "From [t1] to [t2], the values were value1, ..., valuen every 10 minutes. The total trend value was Trends",
}

TRAFFIC_INPUT_TEMPLATES = {
    "CA": "From [t1] to [t2], the traffic flow values were value1, ..., valuen every 5 minutes. The total trend value was Trends",
    "GLA": "From [t1] to [t2], the traffic flow values were value1, ..., valuen every 5 minutes. The total trend value was Trends",
    "GBA": "From [t1] to [t2], the traffic flow values were value1, ..., valuen every 5 minutes. The total trend value was Trends",
    "SD": "From [t1] to [t2], the traffic flow values were value1, ..., valuen every 5 minutes. The total trend value was Trends",
    "SACRAMENTO": "From [t1] to [t2], the traffic flow values were value1, ..., valuen every 5 minutes. The total trend value was Trends",
    "SACRA": "From [t1] to [t2], the traffic flow values were value1, ..., valuen every 5 minutes. The total trend value was Trends",
    "XTRAFFIC": "From [t1] to [t2], the traffic flow values were value1, ..., valuen every 5 minutes. The total trend value was Trends",
}

DATE_ONLY_DATASETS = {"FRED", "ILI"}
HOUR_ONLY_DATASETS = {"ETTh1", "ETTh2", "ECL"}


class TimeCMA(BaseModel):
    """
    LargeST-compatible TimeCMA using the official Dual-branch layout.

    The official TimeCMA backbone assumes one scalar series per node. In
    LargeST, the first history channel is treated as that forecasting signal,
    while optional prompt channels or external prompt embeddings feed the
    prompt branch.
    """

    def __init__(
        self,
        ts_dim,
        prompt_dim,
        channel,
        prompt_hidden,
        e_layer,
        d_layer,
        d_ff,
        head,
        dropout,
        prompt_pool,
        external_prompt_dim,
        prompt_gen_model_name,
        prompt_gen_local_files_only,
        prompt_gen_allow_download,
        prompt_max_tokens,
        prompt_data_name="Traffic",
        prompt_input_template="",
        prompt_freq_minutes=5,
        **args
    ):
        super(TimeCMA, self).__init__(**args)

        if ts_dim + prompt_dim != self.input_dim:
            raise ValueError(
                "ts_dim + prompt_dim must equal input_dim, got {} + {} != {}".format(
                    ts_dim, prompt_dim, self.input_dim
                )
            )
        if ts_dim < 1:
            raise ValueError("ts_dim must be >= 1")
        if channel % head != 0:
            raise ValueError("channel must be divisible by head")
        if external_prompt_dim % head != 0:
            raise ValueError("external_prompt_dim must be divisible by head")

        self.ts_dim = ts_dim
        self.prompt_dim = prompt_dim
        self.channel = channel
        self.prompt_hidden = prompt_hidden
        self.prompt_pool = prompt_pool
        self.external_prompt_dim = external_prompt_dim
        self.prompt_gen_model_name = prompt_gen_model_name
        self.prompt_gen_local_files_only = bool(prompt_gen_local_files_only)
        self.prompt_gen_allow_download = bool(prompt_gen_allow_download)
        self.prompt_max_tokens = prompt_max_tokens
        self.prompt_data_name = str(prompt_data_name)
        self.prompt_input_template = str(prompt_input_template) if prompt_input_template is not None else ""
        self.prompt_freq_minutes = int(prompt_freq_minutes)

        self.normalize_layer = Normalize(self.node_num, affine=False)
        self.length_to_feature = nn.Linear(self.seq_len, self.channel)

        encoder_args = dict(
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )
        self.ts_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=channel,
                nhead=head,
                **encoder_args
            ),
            num_layers=e_layer,
        )

        self.prompt_missing = nn.Parameter(
            torch.zeros(1, self.node_num, self.external_prompt_dim)
        )
        self.prompt_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.external_prompt_dim,
                nhead=head,
                **encoder_args
            ),
            num_layers=e_layer,
        )
        if prompt_dim > 0:
            self.prompt_proj = nn.Linear(prompt_dim, self.external_prompt_dim)
        else:
            self.prompt_proj = None

        self.prompt_stat_projector = nn.Sequential(
            nn.Linear(max(4, ts_dim * 2), self.prompt_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.prompt_hidden, self.external_prompt_dim),
        )
        self._prompt_generator_model = None
        self._prompt_generator_tokenizer = None

        self.cross = CrossModal(
            d_model=self.node_num,
            n_heads=1,
            d_ff=d_ff,
            norm="LayerNorm",
            attn_dropout=dropout,
            dropout=dropout,
            pre_norm=True,
            activation="gelu",
            res_attention=True,
            n_layers=1,
            store_attn=False,
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=channel,
                nhead=head,
                batch_first=True,
                norm_first=True,
                dropout=dropout,
            ),
            num_layers=d_layer,
        )

        self.projection = nn.Linear(channel, self.horizon * self.output_dim, bias=True)

    def _ensure_prompt_generator(self, device):
        if self._prompt_generator_model is not None and self._prompt_generator_tokenizer is not None:
            self._prompt_generator_model = self._prompt_generator_model.to(device)
            return
        if GPT2Model is None or GPT2Tokenizer is None:
            raise ImportError(
                "transformers is required for GPT2 prompt embedding generation. "
                "Install transformers or use method='stats'."
            )

        try:
            tokenizer = GPT2Tokenizer.from_pretrained(
                self.prompt_gen_model_name,
                local_files_only=self.prompt_gen_local_files_only,
            )
            model = GPT2Model.from_pretrained(
                self.prompt_gen_model_name,
                local_files_only=self.prompt_gen_local_files_only,
            )
        except Exception:
            if not self.prompt_gen_allow_download:
                raise
            tokenizer = GPT2Tokenizer.from_pretrained(
                self.prompt_gen_model_name,
                local_files_only=False,
            )
            model = GPT2Model.from_pretrained(
                self.prompt_gen_model_name,
                local_files_only=False,
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        self._prompt_generator_model = model
        self._prompt_generator_tokenizer = tokenizer

    def _minutes_to_phrase(self, freq_minutes):
        if freq_minutes <= 0:
            return "step"
        if freq_minutes % (24 * 60) == 0:
            days = freq_minutes // (24 * 60)
            return "day" if days == 1 else "{} days".format(days)
        if freq_minutes % 60 == 0:
            hours = freq_minutes // 60
            return "hour" if hours == 1 else "{} hours".format(hours)
        return "{} minutes".format(freq_minutes)

    def _resolve_input_template(self, data_name=None, input_template=None):
        if input_template is None:
            input_template = self.prompt_input_template
        if input_template is not None and str(input_template).strip():
            return str(input_template).strip()

        key = str(data_name if data_name is not None else self.prompt_data_name).strip()
        if key in OFFICIAL_INPUT_TEMPLATES:
            return OFFICIAL_INPUT_TEMPLATES[key]
        for name, template in OFFICIAL_INPUT_TEMPLATES.items():
            if key.lower() == name.lower():
                return template

        key_upper = key.upper()
        if key_upper in TRAFFIC_INPUT_TEMPLATES:
            return TRAFFIC_INPUT_TEMPLATES[key_upper]

        freq_phrase = self._minutes_to_phrase(self.prompt_freq_minutes)
        return (
            "From [t1] to [t2], the values were value1, ..., valuen every {}. "
            "The total trend value was Trends"
        ).format(freq_phrase)

    def _format_prompt_date(self, mark_row, data_name):
        values = mark_row.detach().cpu().tolist() if torch.is_tensor(mark_row) else list(mark_row)
        if len(values) < 3:
            raise ValueError("input_mark row must include at least year/month/day, got {}".format(values))

        year = int(values[0])
        month = int(values[1])
        day = int(values[2])
        hour = int(values[4]) if len(values) > 4 else 0
        minute = int(values[5]) if len(values) > 5 else 0

        key = str(data_name if data_name is not None else self.prompt_data_name).strip()
        key_upper = key.upper()
        if key in DATE_ONLY_DATASETS or key_upper in DATE_ONLY_DATASETS:
            return "{:02d}/{:02d}/{:04d}".format(day, month, year)
        if key in HOUR_ONLY_DATASETS or key_upper in HOUR_ONLY_DATASETS:
            return "{:02d}/{:02d}/{:04d} {:02d}:00".format(day, month, year, hour)
        return "{:02d}/{:02d}/{:04d} {:02d}:{:02d}".format(day, month, year, hour, minute)

    def generate_prompt_embeddings(
        self,
        ts_inputs,
        input_mark=None,
        method="stats",
        data_name=None,
        input_template=None,
    ):
        """
        Generate prompt embeddings for external-embedding TimeCMA mode.

        Args:
            ts_inputs: [B, T, N, ts_dim] or [B, T, N]
            input_mark: timestamp features [B, T, M], M>=3.
            method: "gpt2" (text prompt last-token embedding) or "stats" (lightweight projector).

        Returns:
            Tensor shaped [B, external_prompt_dim, N, 1]
        """
        if ts_inputs.dim() == 3:
            ts_inputs = ts_inputs.unsqueeze(-1)
        if ts_inputs.dim() != 4:
            raise ValueError("ts_inputs must be 4D [B, T, N, C], got {}".format(ts_inputs.shape))

        batch_size, seq_len, node_num, feat_dim = ts_inputs.shape
        if feat_dim < 1:
            raise ValueError("ts_inputs must include at least one feature channel")

        base_series = ts_inputs[..., 0]
        min_v = base_series.min(dim=1).values
        max_v = base_series.max(dim=1).values
        mean_v = base_series.mean(dim=1)
        trend_v = (base_series[:, -1, :] - base_series[:, 0, :])

        if method == "stats":
            # [B, N, 4]
            stats = torch.stack([min_v, max_v, mean_v, trend_v], dim=-1)
            stats = stats.reshape(batch_size * node_num, -1)

            if stats.shape[-1] < max(4, self.ts_dim * 2):
                pad_dim = max(4, self.ts_dim * 2) - stats.shape[-1]
                pad = torch.zeros(stats.shape[0], pad_dim, device=stats.device, dtype=stats.dtype)
                stats = torch.cat([stats, pad], dim=-1)
            elif stats.shape[-1] > max(4, self.ts_dim * 2):
                stats = stats[:, : max(4, self.ts_dim * 2)]

            emb = self.prompt_stat_projector(stats)
            emb = emb.view(batch_size, node_num, self.external_prompt_dim)
            emb = emb.permute(0, 2, 1).unsqueeze(-1).contiguous()
            return emb

        if method != "gpt2":
            raise ValueError("Unsupported embedding generation method: {}".format(method))

        if input_mark is None:
            raise ValueError(
                "input_mark is required for method='gpt2'. "
                "Provide [year,month,day,weekday,hour,minute] marks per step."
            )
        if not torch.is_tensor(input_mark):
            input_mark = torch.as_tensor(input_mark, device=ts_inputs.device)
        else:
            input_mark = input_mark.to(ts_inputs.device)
        if input_mark.dim() != 3:
            raise ValueError("input_mark must be 3D [B, T, M], got {}".format(tuple(input_mark.shape)))
        if input_mark.shape[0] != batch_size or input_mark.shape[1] != seq_len:
            raise ValueError(
                "input_mark shape mismatch: expected [{}, {}, M], got {}".format(
                    batch_size, seq_len, tuple(input_mark.shape)
                )
            )

        self._ensure_prompt_generator(ts_inputs.device)
        tokenizer = self._prompt_generator_tokenizer
        model = self._prompt_generator_model
        resolved_data_name = data_name if data_name is not None else self.prompt_data_name
        resolved_template = self._resolve_input_template(
            data_name=resolved_data_name, input_template=input_template
        )

        tokenized_prompts = []
        max_token_count = 0
        for b in range(batch_size):
            for n in range(node_num):
                values = ts_inputs[b, :, n, 0].detach().reshape(-1)
                values_str = ", ".join([str(int(v.item())) for v in values])
                trends = torch.sum(torch.diff(values))
                trends_str = "{:0f}".format(float(trends.item()))

                start_date = self._format_prompt_date(input_mark[b, 0], resolved_data_name)
                end_date = self._format_prompt_date(input_mark[b, seq_len - 1], resolved_data_name)

                prompt = resolved_template.replace("value1, ..., valuen", values_str)
                prompt = prompt.replace("Trends", trends_str)
                prompt = prompt.replace("[t1]", start_date).replace("[t2]", end_date)

                tokenized_prompt = tokenizer.encode(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.prompt_max_tokens,
                ).to(ts_inputs.device)
                max_token_count = max(max_token_count, int(tokenized_prompt.shape[1]))
                tokenized_prompts.append((b, n, tokenized_prompt))

        in_prompt_emb = torch.zeros(
            (batch_size, max_token_count, self.external_prompt_dim, node_num),
            dtype=torch.float32,
            device=ts_inputs.device,
        )

        with torch.no_grad():
            for b, n, tokenized_prompt in tokenized_prompts:
                prompt_embeddings = model(tokenized_prompt).last_hidden_state
                emb_dim = int(prompt_embeddings.shape[-1])
                if emb_dim > self.external_prompt_dim:
                    prompt_embeddings = prompt_embeddings[..., : self.external_prompt_dim]
                elif emb_dim < self.external_prompt_dim:
                    pad_dim = self.external_prompt_dim - emb_dim
                    emb_pad = torch.zeros(
                        prompt_embeddings.shape[0],
                        prompt_embeddings.shape[1],
                        pad_dim,
                        dtype=prompt_embeddings.dtype,
                        device=prompt_embeddings.device,
                    )
                    prompt_embeddings = torch.cat([prompt_embeddings, emb_pad], dim=-1)

                padding_length = max_token_count - int(tokenized_prompt.shape[1])
                if padding_length > 0:
                    last_token_embedding = prompt_embeddings[:, -1:, :]
                    padding = last_token_embedding.repeat(1, padding_length, 1)
                    prompt_embeddings = torch.cat([prompt_embeddings, padding], dim=1)

                in_prompt_emb[b, :max_token_count, :, n] = prompt_embeddings.squeeze(0)

        last_token_emb = in_prompt_emb[:, max_token_count - 1, :, :]
        return last_token_emb.unsqueeze(-1).contiguous()

    def _pool_prompt(self, prompt_inputs):
        if self.prompt_dim == 0:
            prompt = self.prompt_missing.expand(prompt_inputs.shape[0], -1, -1)
            prompt = self.prompt_encoder(prompt)
            return prompt.permute(0, 2, 1).contiguous()

        if self.prompt_pool == "last":
            prompt = prompt_inputs[:, -1]
        elif self.prompt_pool == "max":
            prompt = prompt_inputs.max(dim=1).values
        else:
            prompt = prompt_inputs.mean(dim=1)

        prompt = self.prompt_proj(prompt)
        prompt = self.prompt_encoder(prompt)
        return prompt.permute(0, 2, 1).contiguous()

    def _encode_external_embeddings(self, embeddings):
        if embeddings.dim() == 4 and embeddings.shape[-1] == 1:
            embeddings = embeddings.squeeze(-1)  # [B, E, N] or [B, N, E]
        if embeddings.dim() != 3:
            raise ValueError("External embeddings must be 3D/4D, got {}".format(embeddings.shape))

        if embeddings.shape[1] == self.node_num:
            emb_bnE = embeddings  # [B, N, E]
        elif embeddings.shape[2] == self.node_num:
            emb_bnE = embeddings.permute(0, 2, 1).contiguous()  # [B, N, E]
        else:
            raise ValueError(
                "External embeddings shape {} does not match node_num {}".format(
                    tuple(embeddings.shape), self.node_num
                )
            )

        if emb_bnE.shape[-1] != self.external_prompt_dim:
            raise ValueError(
                "External embedding dim mismatch: expected {}, got {}".format(
                    self.external_prompt_dim, emb_bnE.shape[-1]
                )
            )

        prompt = self.prompt_encoder(emb_bnE)
        return prompt.permute(0, 2, 1).contiguous()

    def forward(self, inputs, label=None, embeddings=None):
        ts_inputs = inputs[..., : self.ts_dim].float()
        prompt_inputs = inputs[..., self.ts_dim :].float()

        target_history = ts_inputs[..., 0]
        norm_target = self.normalize_layer(target_history, "norm")

        batch_size = norm_target.shape[0]
        ts_tokens = norm_target.permute(0, 2, 1).contiguous()
        ts_tokens = self.length_to_feature(ts_tokens)
        ts_tokens = self.ts_encoder(ts_tokens)
        ts_tokens = ts_tokens.permute(0, 2, 1).contiguous()

        if embeddings is not None:
            prompt_tokens = self._encode_external_embeddings(embeddings)
        else:
            prompt_tokens = self._pool_prompt(prompt_inputs)
        cross_tokens = self.cross(ts_tokens, prompt_tokens, prompt_tokens)
        decoded = self.decoder(
            cross_tokens.permute(0, 2, 1).contiguous(),
            cross_tokens.permute(0, 2, 1).contiguous(),
        )

        output = self.projection(decoded)
        output = output.view(batch_size, self.node_num, self.horizon, self.output_dim)
        output = output.permute(0, 2, 1, 3).contiguous()
        output_view = output.permute(0, 1, 3, 2).reshape(
            batch_size, self.horizon * self.output_dim, self.node_num
        )
        output_view = self.normalize_layer(output_view, "denorm")
        output = output_view.view(batch_size, self.horizon, self.output_dim, self.node_num)
        output = output.permute(0, 1, 3, 2).contiguous()
        return output


class Normalize(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class CrossModal(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        d_ff=None,
        norm="LayerNorm",
        attn_dropout=0.0,
        dropout=0.0,
        activation="gelu",
        res_attention=False,
        n_layers=1,
        pre_norm=False,
        store_attn=False,
    ):
        super(CrossModal, self).__init__()
        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    d_model,
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=d_ff,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=activation,
                    res_attention=res_attention,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                )
                for _ in range(n_layers)
            ]
        )
        self.res_attention = res_attention

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        scores = None
        if self.res_attention:
            for mod in self.layers:
                q, scores = mod(q, k, v, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return q

        for mod in self.layers:
            q = mod(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return q


class TSTEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        d_ff=256,
        store_attn=False,
        norm="LayerNorm",
        attn_dropout=0.0,
        dropout=0.0,
        bias=True,
        activation="gelu",
        res_attention=False,
        pre_norm=False,
    ):
        super(TSTEncoderLayer, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(
            d_model,
            n_heads,
            d_k=d_k,
            d_v=d_v,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention,
        )

        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, q, k, v, prev=None, key_padding_mask=None, attn_mask=None):
        if self.pre_norm:
            q = self.norm_attn(q)
            k = self.norm_attn(k)
            v = self.norm_attn(v)

        if self.res_attention:
            q2, attn, scores = self.self_attn(q, k, v, prev, key_padding_mask, attn_mask)
        else:
            q2, attn = self.self_attn(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        if self.store_attn:
            self.attn = attn

        q = q + self.dropout_attn(q2)
        if not self.pre_norm:
            q = self.norm_attn(q)

        if self.pre_norm:
            q = self.norm_ffn(q)
        q2 = self.ff(q)
        q = q + self.dropout_ffn(q2)
        if not self.pre_norm:
            q = self.norm_ffn(q)

        if self.res_attention:
            return q, scores
        return q


class _MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        res_attention=False,
        attn_dropout=0.0,
        proj_dropout=0.0,
        qkv_bias=True,
    ):
        super(_MultiheadAttention, self).__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(
            d_model,
            n_heads,
            attn_dropout=attn_dropout,
            res_attention=res_attention,
        )
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q, K=None, V=None, prev=None, key_padding_mask=None, attn_mask=None):
        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q

        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_s,
                k_s,
                v_s,
                prev=prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            output, attn_weights = self.sdp_attn(
                q_s,
                k_s,
                v_s,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )

        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0.0, res_attention=False):
        super(_ScaledDotProductAttention, self).__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=False)

    def forward(self, q, k, v, prev=None, key_padding_mask=None, attn_mask=None):
        attn_scores = torch.matmul(q, k) * self.scale
        if prev is not None:
            attn_scores = attn_scores + prev
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_scores = attn_scores + attn_mask
        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)

        if self.res_attention:
            return output, attn_weights, attn_scores
        return output, attn_weights


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    if activation.lower() == "relu":
        return nn.ReLU()
    if activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError("{} is not available".format(activation))
