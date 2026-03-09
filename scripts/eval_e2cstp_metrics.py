#!/usr/bin/env python3
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.e2cstp.main import (  # noqa: E402
    _bool_flag,
    _build_node_prompts,
    _encode_prompts_with_gpt2,
    _read_meta,
    _resolve_node_order_path,
    _resolve_text_emb_cache_path,
    set_seed,
)
from src.models.e2cstp import E2CSTP  # noqa: E402
from src.utils.dataloader import (  # noqa: E402
    get_dataset_info,
    load_adj_from_numpy,
    load_dataset,
    load_dataset_for_e2cstp_static_text,
)
from src.utils.graph_algo import normalize_adj_mx  # noqa: E402


def _load_state_dict(ckpt_path, device):
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


def _build_dataloader(args):
    if args.data_path and args.adj_path and args.node_num > 0:
        data_path = args.data_path
        adj_path = args.adj_path
        node_num = args.node_num
    else:
        data_path, adj_path, node_num = get_dataset_info(args.dataset)

    if _bool_flag(args.use_meta_text_prompt):
        node_order_path = _resolve_node_order_path(args, data_path)
        node_order = np.load(node_order_path)
        if len(node_order) != int(node_num):
            raise ValueError(
                "node_order length mismatch: {} vs node_num {}".format(len(node_order), node_num)
            )

        text_emb_cache_path = _resolve_text_emb_cache_path(args)
        if os.path.exists(text_emb_cache_path):
            text_embeddings = np.load(text_emb_cache_path)
        else:
            meta = _read_meta(args.meta_path)
            prompts = _build_node_prompts(meta, node_order, args.meta_prompt_prefix)
            text_embeddings = _encode_prompts_with_gpt2(prompts, args, torch.device(args.device), _DummyLogger())
            os.makedirs(os.path.dirname(text_emb_cache_path), exist_ok=True)
            np.save(text_emb_cache_path, text_embeddings)

        if text_embeddings.shape != (int(node_num), int(args.text_dim)):
            raise ValueError(
                "text embedding shape mismatch, expected ({}, {}), got {}".format(
                    node_num, args.text_dim, text_embeddings.shape
                )
            )
        dataloader, scaler = load_dataset_for_e2cstp_static_text(
            data_path, args, _DummyLogger(), text_embeddings
        )
    else:
        dataloader, scaler = load_dataset(data_path, args, _DummyLogger())

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, args.adj_type)
    supports = [torch.tensor(adj, dtype=torch.float32, device=torch.device(args.device)) for adj in adj_mx]
    return dataloader, scaler, supports, node_num


class _DummyLogger:
    def info(self, *_args, **_kwargs):
        return None


def _build_model(args, supports, node_num):
    return E2CSTP(
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


def _find_mask_value(test_loader, scaler, device):
    min_value = None
    with torch.no_grad():
        for _, label in test_loader.get_iterator():
            label = torch.tensor(label, dtype=torch.float32, device=device)
            label = scaler.inverse_transform(label)
            current = label.min().item()
            if min_value is None or current < min_value:
                min_value = current
    if min_value is None:
        raise RuntimeError("Empty test loader")
    if min_value < 1:
        return min_value
    return 0.0


def _evaluate_metrics(model, test_loader, scaler, device, mask_value):
    total_abs = 0.0
    total_sq = 0.0
    total_count = 0.0
    total_mape = 0.0
    total_smape = 0.0
    total_label_abs = 0.0

    with torch.no_grad():
        for X, label in test_loader.get_iterator():
            X = torch.tensor(X, dtype=torch.float32, device=device)
            label = torch.tensor(label, dtype=torch.float32, device=device)

            pred = model(X, label)["prediction"]
            pred = scaler.inverse_transform(pred)
            label = scaler.inverse_transform(label)

            valid = label != mask_value
            if not valid.any():
                continue

            pred_v = pred[valid]
            label_v = label[valid]
            abs_err = torch.abs(pred_v - label_v)

            total_abs += abs_err.sum().item()
            total_sq += torch.square(pred_v - label_v).sum().item()
            total_count += float(valid.sum().item())

            mape_term = abs_err / torch.clamp(torch.abs(label_v), min=1e-6)
            total_mape += mape_term.sum().item()

            smape_term = 2.0 * abs_err / torch.clamp(torch.abs(pred_v) + torch.abs(label_v), min=1e-6)
            total_smape += smape_term.sum().item()

            total_label_abs += torch.abs(label_v).sum().item()

    if total_count == 0:
        raise RuntimeError("No valid elements to evaluate")

    mae = total_abs / total_count
    rmse = (total_sq / total_count) ** 0.5
    mape = total_mape / total_count
    smape = total_smape / total_count
    wape = total_abs / max(total_label_abs, 1e-6)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_pct": mape * 100.0,
        "WAPE_pct": wape * 100.0,
        "sMAPE_pct": smape * 100.0,
    }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="SD")
    parser.add_argument("--years", type=str, default="2023")
    parser.add_argument("--model_name", type=str, default="e2cstp")
    parser.add_argument("--seed", type=int, default=2023)

    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--input_dim", type=int, default=33)
    parser.add_argument("--output_dim", type=int, default=1)

    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--adj_type", type=str, default="doubletransition")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--adj_path", type=str, default="")
    parser.add_argument("--node_num", type=int, default=0)
    parser.add_argument("--run_tag", type=str, default="")

    parser.add_argument("--st_dim", type=int, default=1)
    parser.add_argument("--text_dim", type=int, default=32)
    parser.add_argument("--image_dim", type=int, default=0)
    parser.add_argument("--use_meta_text_prompt", type=int, default=1)
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
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--save_json", type=str, default="")
    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    dataloader, scaler, supports, node_num = _build_dataloader(args)
    model = _build_model(args, supports, node_num).to(device)
    model.eval()

    if args.checkpoint_path:
        ckpt_path = args.checkpoint_path
    else:
        folder_parts = [
            str(args.dataset),
            str(args.st_dim),
            str(args.text_dim),
            str(args.image_dim),
            "q{}".format(args.seq_len),
            "h{}".format(args.horizon),
        ]
        run_tag = str(args.run_tag).strip()
        if run_tag:
            folder_parts.append(run_tag.replace("/", "_").replace(" ", "_"))
        folder_name = "-".join(folder_parts)
        ckpt_path = os.path.join(
            "/root/LargeST/experiments", args.model_name, folder_name, "final_model_s{}.pt".format(args.seed)
        )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("checkpoint not found: {}".format(ckpt_path))

    state_dict = _load_state_dict(ckpt_path, device)
    model.load_state_dict(state_dict)

    test_loader = dataloader["test_loader"]
    mask_value = _find_mask_value(test_loader, scaler, device)
    metrics = _evaluate_metrics(model, test_loader, scaler, device, mask_value)

    print(
        "MAE={:.4f} RMSE={:.4f} MAPE={:.2f}% WAPE={:.2f}% sMAPE={:.2f}%".format(
            metrics["MAE"],
            metrics["RMSE"],
            metrics["MAPE_pct"],
            metrics["WAPE_pct"],
            metrics["sMAPE_pct"],
        )
    )
    print("mask_value={:.8f}".format(mask_value))
    print("checkpoint={}".format(ckpt_path))

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": ckpt_path,
                    "mask_value": mask_value,
                    "metrics": metrics,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


if __name__ == "__main__":
    main()
