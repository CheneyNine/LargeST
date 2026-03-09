#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime
from datetime import timedelta

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.timecma import TimeCMA  # noqa: E402
from src.utils.dataloader import get_dataset_info  # noqa: E402
from src.utils.dataloader import load_dataset  # noqa: E402
from src.utils.dataloader import load_dataset_with_embeddings  # noqa: E402


class _DummyLogger:
    def info(self, *_args, **_kwargs):
        return None


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def _parse_prompt_start_datetime(start_datetime):
    value = str(start_datetime).strip()
    if not value:
        return datetime(2019, 1, 1, 0, 0, 0)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(value, fmt)
            if fmt == "%Y-%m-%d":
                return parsed.replace(hour=0, minute=0, second=0)
            return parsed
        except ValueError:
            continue
    raise ValueError(
        "Invalid prompt_start_datetime: {}. Use one of YYYY-MM-DD, YYYY-MM-DD HH:MM:SS, YYYY-MM-DDTHH:MM:SS".format(
            value
        )
    )


def _load_state_dict(ckpt_path, device):
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


def _bool_flag(value):
    return bool(int(value)) if isinstance(value, (int, np.integer, str)) else bool(value)


def _build_dataloader(args):
    if str(args.data_path).strip():
        data_path = args.data_path
        if args.node_num > 0:
            node_num = int(args.node_num)
        else:
            ptr = np.load(os.path.join(data_path, args.years, "his.npz"))
            node_num = int(ptr["data"].shape[1])
    else:
        data_path, _, node_num = get_dataset_info(args.dataset)

    if _bool_flag(args.use_external_embeddings) or _bool_flag(args.generate_embeddings_on_the_fly):
        dataloader, scaler = load_dataset_with_embeddings(
            data_path, args, _DummyLogger(), embedding_prefix=args.embedding_prefix
        )
    else:
        dataloader, scaler = load_dataset(data_path, args, _DummyLogger())
    return dataloader, scaler, data_path, node_num


def _build_model(args, node_num):
    return TimeCMA(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        ts_dim=args.ts_dim,
        prompt_dim=args.prompt_dim,
        channel=args.channel,
        prompt_hidden=args.prompt_hidden,
        e_layer=args.e_layer,
        d_layer=args.d_layer,
        d_ff=args.d_ff,
        head=args.head,
        dropout=args.dropout,
        prompt_pool=args.prompt_pool,
        external_prompt_dim=args.external_prompt_dim,
        prompt_gen_model_name=args.prompt_gen_model_name,
        prompt_gen_local_files_only=args.prompt_gen_local_files_only,
        prompt_gen_allow_download=args.prompt_gen_allow_download,
        prompt_max_tokens=args.prompt_max_tokens,
        prompt_data_name=args.prompt_data_name,
        prompt_input_template=args.prompt_input_template,
        prompt_freq_minutes=args.prompt_freq_minutes,
    )


def _resolve_checkpoint_path(args):
    if str(args.checkpoint_path).strip():
        return args.checkpoint_path

    folder_parts = [
        str(args.dataset),
        "q{}".format(args.seq_len),
        "h{}".format(args.horizon),
        "ts{}".format(args.ts_dim),
        "pd{}".format(args.prompt_dim),
        "c{}".format(args.channel),
        "el{}".format(args.e_layer),
        "dl{}".format(args.d_layer),
        "lr{}".format(args.lrate),
        "drop{}".format(args.dropout),
        "s{}".format(args.seed),
    ]
    run_tag = str(args.run_tag).strip()
    if run_tag:
        folder_parts.append(run_tag)
    folder_name = "-".join(folder_parts)
    return os.path.join(
        "/root/LargeST/experiments",
        args.model_name,
        folder_name,
        "final_model_s{}.pt".format(args.seed),
    )


def _unpack_batch(batch):
    if isinstance(batch, (list, tuple)) and len(batch) == 4:
        return batch[0], batch[1], batch[2], batch[3]
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        return batch[0], batch[1], batch[2], None
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], batch[1], None, None
    raise ValueError("Unexpected batch format: {}".format(type(batch)))


def _build_prompt_marks(idx_ind, seq_len, prompt_start_datetime, prompt_freq_minutes, device):
    if idx_ind is None:
        return None
    idx_ind = np.asarray(idx_ind).reshape(-1)
    x_offsets = np.arange(-(seq_len - 1), 1, 1, dtype=np.int64)
    history_index = idx_ind[:, None] + x_offsets[None, :]
    marks = np.zeros((history_index.shape[0], history_index.shape[1], 6), dtype=np.int64)
    for i in range(history_index.shape[0]):
        for t in range(history_index.shape[1]):
            dt = prompt_start_datetime + timedelta(
                minutes=prompt_freq_minutes * int(history_index[i, t])
            )
            marks[i, t, 0] = dt.year
            marks[i, t, 1] = dt.month
            marks[i, t, 2] = dt.day
            marks[i, t, 3] = dt.weekday()
            marks[i, t, 4] = dt.hour
            marks[i, t, 5] = dt.minute
    return torch.tensor(marks, dtype=torch.float32, device=device)


def _prepare_embeddings(args, model, embeddings, x_tensor, idx_ind, prompt_start_datetime, device):
    if embeddings is not None:
        return torch.tensor(embeddings, dtype=torch.float32, device=device)
    if not _bool_flag(args.generate_embeddings_on_the_fly):
        return None

    input_mark = None
    if args.embedding_method == "gpt2":
        input_mark = _build_prompt_marks(
            idx_ind=idx_ind,
            seq_len=args.seq_len,
            prompt_start_datetime=prompt_start_datetime,
            prompt_freq_minutes=int(args.prompt_freq_minutes),
            device=device,
        )
        with torch.no_grad():
            return model.generate_prompt_embeddings(
                x_tensor[..., : model.ts_dim],
                input_mark=input_mark,
                method=args.embedding_method,
            )
    return model.generate_prompt_embeddings(
        x_tensor[..., : model.ts_dim],
        input_mark=input_mark,
        method=args.embedding_method,
    )


def _compute_mask_value(test_loader, scaler, device):
    min_value = None
    with torch.no_grad():
        for batch in test_loader.get_iterator():
            _, label, _, _ = _unpack_batch(batch)
            label = torch.tensor(label, dtype=torch.float32, device=device)
            label = scaler.inverse_transform(label)
            current = label.min().item()
            if min_value is None or current < min_value:
                min_value = current
    if min_value is None:
        raise RuntimeError("Empty test loader")
    return min_value if min_value < 1 else 0.0


def _evaluate_metrics(model, test_loader, scaler, args, device):
    prompt_start_datetime = _parse_prompt_start_datetime(args.prompt_start_datetime)
    mask_value = _compute_mask_value(test_loader, scaler, device)

    total_abs = 0.0
    total_sq = 0.0
    total_count = 0.0
    total_mape = 0.0
    total_smape = 0.0
    total_label_abs = 0.0

    with torch.no_grad():
        for batch in test_loader.get_iterator():
            X, label, embeddings, idx_ind = _unpack_batch(batch)
            X = torch.tensor(X, dtype=torch.float32, device=device)
            label = torch.tensor(label, dtype=torch.float32, device=device)
            embedding_tensor = _prepare_embeddings(
                args=args,
                model=model,
                embeddings=embeddings,
                x_tensor=X,
                idx_ind=idx_ind,
                prompt_start_datetime=prompt_start_datetime,
                device=device,
            )

            pred = model(X, label, embeddings=embedding_tensor)
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
            total_label_abs += torch.abs(label_v).sum().item()

            mape_term = abs_err / torch.clamp(torch.abs(label_v), min=1e-6)
            total_mape += mape_term.sum().item()

            smape_term = 2.0 * abs_err / torch.clamp(torch.abs(pred_v) + torch.abs(label_v), min=1e-6)
            total_smape += smape_term.sum().item()

    if total_count == 0:
        raise RuntimeError("No valid elements to evaluate")

    metrics = {
        "MAE": total_abs / total_count,
        "RMSE": (total_sq / total_count) ** 0.5,
        "MAPE_pct": (total_mape / total_count) * 100.0,
        "WAPE_pct": (total_abs / max(total_label_abs, 1e-6)) * 100.0,
        "sMAPE_pct": (total_smape / total_count) * 100.0,
    }
    return mask_value, metrics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="SD")
    parser.add_argument("--years", type=str, default="2019")
    parser.add_argument("--model_name", type=str, default="timecma")
    parser.add_argument("--seed", type=int, default=2024)

    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--output_dim", type=int, default=1)

    parser.add_argument("--ts_dim", type=int, default=3)
    parser.add_argument("--prompt_dim", type=int, default=0)
    parser.add_argument("--channel", type=int, default=32)
    parser.add_argument("--prompt_hidden", type=int, default=128)
    parser.add_argument("--e_layer", type=int, default=1)
    parser.add_argument("--d_layer", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=32)
    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--prompt_pool", type=str, default="mean")
    parser.add_argument("--d_llm", type=int, default=768)
    parser.add_argument("--external_prompt_dim", type=int, default=768)
    parser.add_argument("--prompt_gen_model_name", type=str, default="gpt2")
    parser.add_argument("--prompt_gen_local_files_only", type=int, default=1)
    parser.add_argument("--prompt_gen_allow_download", type=int, default=1)
    parser.add_argument("--prompt_max_tokens", type=int, default=512)
    parser.add_argument("--prompt_data_name", type=str, default="")
    parser.add_argument("--prompt_input_template", type=str, default="")
    parser.add_argument("--prompt_freq_minutes", type=int, default=5)
    parser.add_argument("--prompt_start_datetime", type=str, default="")
    parser.add_argument("--embedding_prefix", type=str, default="prompt_emb")
    parser.add_argument("--generate_embeddings_on_the_fly", type=int, default=0)
    parser.add_argument("--use_external_embeddings", type=int, default=0)
    parser.add_argument("--embedding_method", type=str, default="gpt2", choices=["gpt2", "stats"])

    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--node_num", type=int, default=-1)
    parser.add_argument("--lrate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--save_json", type=str, default="")
    return parser.parse_args()


def main():
    args = get_args()
    if int(args.external_prompt_dim) != int(args.d_llm):
        raise ValueError(
            "external_prompt_dim must equal d_llm for official-style TimeCMA, got {} vs {}".format(
                args.external_prompt_dim, args.d_llm
            )
        )
    if not str(args.prompt_data_name).strip():
        args.prompt_data_name = args.dataset
    if not str(args.prompt_start_datetime).strip():
        year_token = str(args.years).split("_")[0]
        if year_token.isdigit() and len(year_token) == 4:
            args.prompt_start_datetime = "{}-01-01 00:00:00".format(year_token)
        else:
            args.prompt_start_datetime = "2019-01-01 00:00:00"

    set_seed(args.seed)
    device = torch.device(args.device)

    dataloader, scaler, data_path, node_num = _build_dataloader(args)
    model = _build_model(args, node_num).to(device)
    ckpt_path = _resolve_checkpoint_path(args)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("checkpoint not found: {}".format(ckpt_path))

    state_dict = _load_state_dict(ckpt_path, device)
    model.load_state_dict(state_dict)
    model.eval()

    mask_value, metrics = _evaluate_metrics(
        model=model,
        test_loader=dataloader["test_loader"],
        scaler=scaler,
        args=args,
        device=device,
    )

    print("data_path={}".format(data_path))
    print("checkpoint={}".format(ckpt_path))
    print("mask_value={:.8f}".format(mask_value))
    print(
        "MAE={:.4f} RMSE={:.4f} MAPE={:.2f}% WAPE={:.2f}% sMAPE={:.2f}%".format(
            metrics["MAE"],
            metrics["RMSE"],
            metrics["MAPE_pct"],
            metrics["WAPE_pct"],
            metrics["sMAPE_pct"],
        )
    )

    if args.save_json:
        output_dir = os.path.dirname(args.save_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "data_path": data_path,
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
