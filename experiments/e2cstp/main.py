import os
import numpy as np
import pandas as pd

import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

import torch
try:
    from transformers import GPT2Tokenizer, GPT2Model
except Exception:
    GPT2Tokenizer = None
    GPT2Model = None

torch.set_num_threads(3)

from src.engines.e2cstp_engine import E2CSTP_Engine
from src.models.e2cstp import E2CSTP
from src.utils.args import get_public_config
from src.utils.dataloader import get_dataset_info
from src.utils.dataloader import load_adj_from_numpy
from src.utils.dataloader import load_dataset
from src.utils.dataloader import load_dataset_for_e2cstp_static_text
from src.utils.experiment_naming import build_experiment_dir_name
from src.utils.graph_algo import normalize_adj_mx
from src.utils.logging import get_logger
from src.utils.metrics import masked_mae
from src.utils.swanlab_tracker import resolve_swanlab_job_type


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def _bool_flag(value):
    return bool(int(value))


def _read_meta(meta_path):
    meta = pd.read_csv(meta_path, sep="\t", engine="python", on_bad_lines="skip")
    if "station_id" not in meta.columns:
        meta = pd.read_csv(meta_path, low_memory=False)
    if "station_id" not in meta.columns:
        raise ValueError("meta file must contain station_id column: {}".format(meta_path))

    meta = meta.copy()
    meta["station_id"] = pd.to_numeric(meta["station_id"], errors="coerce")
    meta = meta.dropna(subset=["station_id"]).copy()
    meta["station_id"] = meta["station_id"].astype(np.int64)
    return meta


def _format_meta_value(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    if lower in {"nan", "none", "null"}:
        return None
    return text


def _build_node_prompts(meta, node_order, prompt_prefix):
    meta_cols = [col for col in meta.columns if col != "station_id"]
    row_map = {int(row["station_id"]): row for _, row in meta.iterrows()}

    prompts = []
    for sid in node_order.astype(np.int64).tolist():
        row = row_map.get(int(sid))
        if row is None:
            prompts.append("{} station_id={} ; metadata unknown.".format(prompt_prefix, int(sid)))
            continue

        parts = []
        for col in meta_cols:
            value = _format_meta_value(row[col])
            if value is not None:
                parts.append("{}={}".format(col, value))
        if not parts:
            prompts.append("{} station_id={} ; metadata unknown.".format(prompt_prefix, int(sid)))
        else:
            prompts.append("{} station_id={} ; {}.".format(prompt_prefix, int(sid), "; ".join(parts)))
    return prompts


def _load_prompt_llm(args, device):
    if GPT2Tokenizer is None or GPT2Model is None:
        raise ImportError("transformers is required for GPT2 prompt encoding")

    local_files_only = _bool_flag(args.meta_prompt_local_files_only)
    allow_download = _bool_flag(args.meta_prompt_allow_download)
    model_name = args.meta_prompt_model_name

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        model = GPT2Model.from_pretrained(model_name, local_files_only=local_files_only)
    except Exception:
        if not allow_download:
            raise
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, local_files_only=False)
        model = GPT2Model.from_pretrained(model_name, local_files_only=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return tokenizer, model


def _encode_prompts_with_gpt2(prompts, args, device, logger):
    tokenizer, llm_model = _load_prompt_llm(args, device)
    batch_size = max(1, int(args.meta_prompt_batch_size))
    max_tokens = max(16, int(args.meta_prompt_max_tokens))

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            end = min(len(prompts), start + batch_size)
            tokenized = tokenizer(
                prompts[start:end],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_tokens,
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            outputs = llm_model(**tokenized).last_hidden_state  # [B, L, E]
            attention_mask = tokenized["attention_mask"]
            last_pos = attention_mask.sum(dim=1) - 1
            gather_index = last_pos.view(-1, 1, 1).expand(-1, 1, outputs.shape[-1])
            last_token = outputs.gather(dim=1, index=gather_index).squeeze(1)
            embeddings.append(last_token.cpu())
            if start == 0:
                logger.info("Prompt encode first batch: prompt_num={}, embed_dim={}".format(end - start, last_token.shape[-1]))

    embeddings = torch.cat(embeddings, dim=0).numpy().astype(np.float32)
    if embeddings.shape[1] > args.text_dim:
        embeddings = embeddings[:, : args.text_dim]
    elif embeddings.shape[1] < args.text_dim:
        pad = np.zeros((embeddings.shape[0], args.text_dim - embeddings.shape[1]), dtype=np.float32)
        embeddings = np.concatenate([embeddings, pad], axis=1)
    return embeddings


def _resolve_node_order_path(args, data_path):
    if args.node_order_path:
        return args.node_order_path
    candidates = [
        os.path.join(data_path, "node_order_sacramento.npy"),
        os.path.join(data_path, "node_order.npy"),
        os.path.join(os.path.dirname(data_path), "node_order.npy"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Cannot find node order file, please set --node_order_path")


def _resolve_text_emb_cache_path(args):
    if args.text_emb_cache_path:
        return args.text_emb_cache_path
    data_root = os.getenv("LARGEST_DATA_ROOT", "/public_data/LargeST_data")
    cache_dir = os.path.join(data_root, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(
        cache_dir,
        "e2cstp_meta_text_emb_{}_{}_d{}.npy".format(
        args.dataset, args.years, args.text_dim
        ),
    )


def get_config():
    parser = get_public_config()
    parser.add_argument("--adj_type", type=str, default="doubletransition")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--adj_path", type=str, default="")
    parser.add_argument("--node_num", type=int, default=0)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--experiment_timestamp", type=str, default="")

    parser.add_argument("--st_dim", type=int, default=3)
    parser.add_argument("--text_dim", type=int, default=0)
    parser.add_argument("--image_dim", type=int, default=0)
    parser.add_argument("--use_meta_text_prompt", type=int, default=0)
    parser.add_argument("--meta_path", type=str, default="/root/XTraffic/data/sensor_meta_feature.csv")
    parser.add_argument("--node_order_path", type=str, default="")
    parser.add_argument("--meta_prompt_prefix", type=str, default="Traffic sensor metadata:")
    parser.add_argument("--meta_prompt_model_name", type=str, default="gpt2")
    parser.add_argument("--meta_prompt_local_files_only", type=int, default=1)
    parser.add_argument("--meta_prompt_allow_download", type=int, default=1)
    parser.add_argument("--meta_prompt_max_tokens", type=int, default=256)
    parser.add_argument("--meta_prompt_batch_size", type=int, default=64)
    parser.add_argument("--text_emb_cache_path", type=str, default="")

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--encoder_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mamba_expand", type=int, default=2)
    parser.add_argument("--mamba_kernel", type=int, default=3)
    parser.add_argument("--gcn_order", type=int, default=2)
    parser.add_argument("--decoder_hidden", type=int, default=256)

    parser.add_argument("--graph_fusion", type=float, default=0.7)
    parser.add_argument("--causal_momentum", type=float, default=0.9)
    parser.add_argument("--causal_update_interval", type=int, default=50)
    parser.add_argument("--intervention_scale", type=float, default=0.5)
    parser.add_argument("--branch_alpha", type=float, default=0.5)
    parser.add_argument("--causal_reg_weight", type=float, default=0.05)

    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip_grad_value", type=float, default=5)
    parser.add_argument("--use_swanlab", type=int, default=1)
    parser.add_argument("--swanlab_project", type=str, default="LargeST")
    parser.add_argument("--swanlab_experiment", type=str, default="")
    parser.add_argument("--swanlab_mode", type=str, default="cloud")
    parser.add_argument("--swanlab_description", type=str, default="")
    parser.add_argument("--desc", dest="swanlab_description", type=str)
    parser.add_argument("--swanlab_lark_webhook_url", type=str, default=os.getenv("SWANLAB_LARK_WEBHOOK_URL", ""))
    parser.add_argument("--swanlab_lark_secret", type=str, default=os.getenv("SWANLAB_LARK_SECRET", ""))
    args = parser.parse_args()

    total_modal_dim = args.st_dim + args.text_dim + args.image_dim
    if total_modal_dim != args.input_dim:
        raise ValueError(
            "input_dim must equal st_dim + text_dim + image_dim, got {} vs {}".format(
                args.input_dim, total_modal_dim
            )
        )
    if _bool_flag(args.use_meta_text_prompt) and args.text_dim <= 0:
        raise ValueError("text_dim must be > 0 when --use_meta_text_prompt=1")

    folder_name = build_experiment_dir_name(
        model_name=args.model_name,
        dataset=args.dataset,
        years=args.years,
        seq_len=args.seq_len,
        horizon=args.horizon,
        seed=args.seed,
        extra_parts=[
            ("st", args.st_dim),
            ("txt", args.text_dim),
            ("img", args.image_dim),
            ("meta", int(_bool_flag(args.use_meta_text_prompt))),
        ],
        run_tag=args.run_tag,
        started_at=str(args.experiment_timestamp).strip() or None,
    )
    log_dir = "./experiments/{}/{}/".format(args.model_name, folder_name)
    logger = get_logger(log_dir, __name__, "record_s{}.log".format(args.seed))
    logger.info(args)

    return args, log_dir, logger, folder_name


def main():
    args, log_dir, logger, folder_name = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.data_path and args.adj_path and args.node_num > 0:
        data_path = args.data_path
        adj_path = args.adj_path
        node_num = args.node_num
        logger.info("Use custom dataset path.")
    else:
        data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info("Adj path: " + adj_path)

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, args.adj_type)
    supports = [torch.tensor(adj, dtype=torch.float32, device=device) for adj in adj_mx]

    if _bool_flag(args.use_meta_text_prompt):
        node_order_path = _resolve_node_order_path(args, data_path)
        node_order = np.load(node_order_path)
        if len(node_order) != int(node_num):
            raise ValueError(
                "node_order length mismatch: {} vs node_num {}".format(len(node_order), node_num)
            )
        logger.info("Node order path: " + node_order_path)

        text_emb_cache_path = _resolve_text_emb_cache_path(args)
        if os.path.exists(text_emb_cache_path):
            text_embeddings = np.load(text_emb_cache_path)
            logger.info("Load cached text embedding from {}".format(text_emb_cache_path))
        else:
            logger.info("Generate node text prompts from meta: {}".format(args.meta_path))
            meta = _read_meta(args.meta_path)
            prompts = _build_node_prompts(meta, node_order, args.meta_prompt_prefix)
            text_embeddings = _encode_prompts_with_gpt2(prompts, args, device, logger)
            os.makedirs(os.path.dirname(text_emb_cache_path), exist_ok=True)
            np.save(text_emb_cache_path, text_embeddings)
            logger.info("Save text embedding to {}".format(text_emb_cache_path))

        if text_embeddings.shape != (int(node_num), int(args.text_dim)):
            raise ValueError(
                "text embedding shape mismatch, expected ({}, {}), got {}".format(
                    node_num, args.text_dim, text_embeddings.shape
                )
            )
        dataloader, scaler = load_dataset_for_e2cstp_static_text(
            data_path, args, logger, text_embeddings
        )
    else:
        dataloader, scaler = load_dataset(data_path, args, logger)

    model = E2CSTP(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        supports=supports,
        st_dim=args.st_dim,
        text_dim=args.text_dim,
        image_dim=args.image_dim,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        num_heads=args.num_heads,
        mamba_expand=args.mamba_expand,
        mamba_kernel=args.mamba_kernel,
        gcn_order=args.gcn_order,
        decoder_hidden=args.decoder_hidden,
        graph_fusion=args.graph_fusion,
        causal_momentum=args.causal_momentum,
        causal_update_interval=args.causal_update_interval,
        intervention_scale=args.intervention_scale,
        dropout=args.dropout,
    )

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = None

    engine = E2CSTP_Engine(
        device=device,
        model=model,
        dataloader=dataloader,
        scaler=scaler,
        sampler=None,
        loss_fn=loss_fn,
        lrate=args.lrate,
        optimizer=optimizer,
        scheduler=scheduler,
        clip_grad_value=args.clip_grad_value,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_dir=log_dir,
        logger=logger,
        seed=args.seed,
        branch_alpha=args.branch_alpha,
        causal_reg_weight=args.causal_reg_weight,
        swanlab_cfg={
            "enabled": bool(args.use_swanlab),
            "project": args.swanlab_project,
            "experiment_name": args.swanlab_experiment
            if args.swanlab_experiment
            else folder_name,
            "description": args.swanlab_description,
            "job_type": resolve_swanlab_job_type(args.mode),
            "mode": args.swanlab_mode,
            "logdir": log_dir,
            "lark_webhook_url": args.swanlab_lark_webhook_url,
            "lark_secret": args.swanlab_lark_secret,
            "config": vars(args),
        },
    )

    try:
        if args.mode == "train":
            engine.train()
        else:
            engine.evaluate(args.mode)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
