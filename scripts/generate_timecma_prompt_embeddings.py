#!/usr/bin/env python3
import argparse
import os
from datetime import datetime
from datetime import timedelta

import numpy as np
import torch

from src.models.timecma import TimeCMA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--years", type=str, default="2019")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--ts_dim", type=int, default=3)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--embedding_prefix", type=str, default="prompt_emb")
    parser.add_argument("--embedding_method", type=str, default="gpt2", choices=["gpt2", "stats"])
    parser.add_argument("--d_llm", type=int, default=768)
    parser.add_argument("--external_prompt_dim", type=int, default=768)
    parser.add_argument("--prompt_gen_model_name", type=str, default="gpt2")
    parser.add_argument("--prompt_gen_local_files_only", type=int, default=1)
    parser.add_argument("--prompt_gen_allow_download", type=int, default=1)
    parser.add_argument("--prompt_max_tokens", type=int, default=512)
    parser.add_argument("--data_name", type=str, default="")
    parser.add_argument("--input_template", type=str, default="")
    parser.add_argument("--freq_minutes", type=int, default=5)
    parser.add_argument("--start_datetime", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def infer_data_name(data_path, override_name):
    if str(override_name).strip():
        return str(override_name).strip()
    base = os.path.basename(os.path.normpath(data_path))
    return base if base else "Traffic"


def infer_start_datetime(years, start_datetime):
    start_datetime = str(start_datetime).strip()
    if start_datetime:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(start_datetime, fmt)
                if fmt == "%Y-%m-%d":
                    return parsed.replace(hour=0, minute=0, second=0)
                return parsed
            except ValueError:
                continue
        raise ValueError(
            "Invalid --start_datetime: {}. Use one of YYYY-MM-DD, YYYY-MM-DD HH:MM:SS, YYYY-MM-DDTHH:MM:SS".format(
                start_datetime
            )
        )

    year_token = str(years).split("_")[0]
    if year_token.isdigit() and len(year_token) == 4:
        return datetime(int(year_token), 1, 1, 0, 0, 0)
    return datetime(2019, 1, 1, 0, 0, 0)


def build_full_time_marks(total_steps, start_dt, freq_minutes):
    marks = np.zeros((total_steps, 6), dtype=np.int32)
    for t in range(total_steps):
        current = start_dt + timedelta(minutes=int(freq_minutes) * t)
        marks[t, 0] = current.year
        marks[t, 1] = current.month
        marks[t, 2] = current.day
        marks[t, 3] = current.weekday()
        marks[t, 4] = current.hour
        marks[t, 5] = current.minute
    return marks


def _select_feature_stats(stat_value, ts_dim):
    arr = np.asarray(stat_value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("Empty statistic array in his.npz")
    if arr.size == 1 and ts_dim > 1:
        arr = np.repeat(arr, ts_dim)
    if arr.size < ts_dim:
        raise ValueError("Statistic array size {} smaller than ts_dim {}".format(arr.size, ts_dim))
    return arr[:ts_dim].reshape(1, 1, ts_dim)


def load_prompt_source_data(ptr, ts_dim):
    data = ptr["data"][..., :ts_dim].astype(np.float32)
    mean = _select_feature_stats(ptr["mean"], ts_dim)
    std = _select_feature_stats(ptr["std"], ts_dim)
    return data * std + mean


def build_timecma_for_generator(args, node_num):
    if int(args.external_prompt_dim) != int(args.d_llm):
        raise ValueError(
            "external_prompt_dim must equal d_llm for official-style TimeCMA, got {} vs {}".format(
                args.external_prompt_dim, args.d_llm
            )
        )
    model = TimeCMA(
        node_num=node_num,
        input_dim=args.ts_dim,
        output_dim=1,
        seq_len=args.seq_len,
        horizon=1,
        ts_dim=args.ts_dim,
        prompt_dim=0,
        channel=32,
        prompt_hidden=128,
        e_layer=1,
        d_layer=1,
        d_ff=32,
        head=8,
        dropout=0.2,
        prompt_pool="mean",
        external_prompt_dim=args.external_prompt_dim,
        prompt_gen_model_name=args.prompt_gen_model_name,
        prompt_gen_local_files_only=args.prompt_gen_local_files_only,
        prompt_gen_allow_download=args.prompt_gen_allow_download,
        prompt_max_tokens=args.prompt_max_tokens,
        prompt_data_name=args.data_name,
        prompt_input_template=args.input_template,
        prompt_freq_minutes=args.freq_minutes,
    )
    return model


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    year_dir = os.path.join(args.data_path, args.years)
    ptr = np.load(os.path.join(year_dir, "his.npz"))
    data = load_prompt_source_data(ptr, args.ts_dim)
    total_steps = int(data.shape[0])
    node_num = int(data.shape[1])
    output_dir = str(args.output_dir).strip() if str(args.output_dir).strip() else year_dir
    os.makedirs(output_dir, exist_ok=True)
    data_name = infer_data_name(args.data_path, args.data_name)
    start_dt = infer_start_datetime(args.years, args.start_datetime)
    full_time_marks = build_full_time_marks(total_steps, start_dt, args.freq_minutes)

    args.data_name = data_name
    print(
        "Prompt generation config: data_name={}, start_datetime={}, freq_minutes={}, prompt_source=raw, output_dir={}".format(
            data_name, start_dt.strftime("%Y-%m-%d %H:%M:%S"), args.freq_minutes, output_dir
        )
    )

    model = build_timecma_for_generator(args, node_num=node_num).to(device)
    model.eval()

    x_offsets = np.arange(-(args.seq_len - 1), 1, 1)

    for split in ["train", "val", "test"]:
        idx = np.load(os.path.join(year_dir, "idx_{}.npy".format(split)))
        output_path = os.path.join(output_dir, "{}_{}.npy".format(args.embedding_prefix, split))
        output = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(idx), args.external_prompt_dim, node_num, 1),
        )

        for start in range(0, len(idx), args.batch_size):
            end = min(len(idx), start + args.batch_size)
            batch_idx = idx[start:end]
            history_index = batch_idx[:, None] + x_offsets[None, :]
            x_batch = data[history_index, :, :]  # [B, T, N, ts_dim]
            x_mark_batch = full_time_marks[history_index]  # [B, T, 6]
            x_tensor = torch.tensor(x_batch, dtype=torch.float32, device=device)
            x_mark_tensor = torch.tensor(x_mark_batch, dtype=torch.long, device=device)

            with torch.no_grad():
                emb = model.generate_prompt_embeddings(
                    x_tensor,
                    input_mark=x_mark_tensor,
                    method=args.embedding_method,
                    data_name=data_name,
                    input_template=args.input_template,
                )
            output[start:end] = emb.detach().cpu().numpy().astype(np.float32)

            if start == 0:
                print(
                    "[{}] first batch embedding shape: {}".format(
                        split, tuple(emb.shape)
                    )
                )

        output.flush()
        print("Saved {} embeddings to {}".format(split, output_path))


if __name__ == "__main__":
    main()
