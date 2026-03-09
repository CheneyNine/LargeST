import argparse
import calendar
import json
import shutil
from pathlib import Path

import numpy as np


AUX_FILES = [
    "adj_matrix_sacramento.npy",
    "dis_matrix_sacramento.npy",
    "node_order_sacramento.npy",
    "selected_node_idx.npy",
    "selected_station_id.npy",
    "sensor_meta_feature_selected.csv",
    "summary.json",
    "county_candidates.json",
    "match_incidents_y2023_sacramento.csv",
]


def parse_windows(windows_arg):
    windows = []
    for token in str(windows_arg).split(","):
        token = token.strip()
        if not token:
            continue
        seq_len_str, horizon_str = token.split(":")
        seq_len = int(seq_len_str)
        horizon = int(horizon_str)
        if seq_len <= 0 or horizon <= 0:
            raise ValueError("Invalid window pair: {}".format(token))
        windows.append((seq_len, horizon))
    if not windows:
        raise ValueError("No valid windows parsed from --windows")
    return windows


def month_step_range(year, month, steps_per_day):
    if month < 1 or month > 12:
        raise ValueError("month must be in [1, 12]")
    start_day = 0
    for prev_month in range(1, month):
        start_day += calendar.monthrange(year, prev_month)[1]
    month_days = calendar.monthrange(year, month)[1]
    start_step = start_day * steps_per_day
    end_step = (start_day + month_days) * steps_per_day
    return start_step, end_step, month_days


def build_idx(num_steps, seq_len, horizon, train_ratio, val_ratio):
    valid_start = seq_len - 1
    valid_end = num_steps - horizon
    if valid_end <= valid_start:
        raise ValueError(
            "Not enough steps for seq_len={} horizon={} num_steps={}".format(
                seq_len, horizon, num_steps
            )
        )

    idx = np.arange(valid_start, valid_end, dtype=np.int64)
    num_samples = len(idx)
    num_train = int(round(num_samples * train_ratio))
    num_val = int(round(num_samples * val_ratio))
    if num_train <= 0 or num_val <= 0 or num_train + num_val >= num_samples:
        raise ValueError(
            "Bad split sizes for num_samples={} train_ratio={} val_ratio={}".format(
                num_samples, train_ratio, val_ratio
            )
        )

    idx_train = idx[:num_train]
    idx_val = idx[num_train : num_train + num_val]
    idx_test = idx[num_train + num_val :]
    return idx_train, idx_val, idx_test


def compute_train_scaler(raw_data, idx_val, seq_len):
    train_cutoff = int(idx_val[0] - seq_len)
    if train_cutoff <= 0:
        raise ValueError("train_cutoff must be positive, got {}".format(train_cutoff))
    train_raw = raw_data[:train_cutoff, :, 0]
    mean = float(train_raw.mean())
    std = float(train_raw.std())
    if std <= 0:
        raise ValueError("Computed std must be > 0, got {}".format(std))
    return mean, std


def save_dataset(output_root, year, normalized_data, mean, std, idx_train, idx_val, idx_test):
    year_dir = output_root / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        year_dir / "his.npz",
        data=normalized_data.astype(np.float32),
        mean=np.array(mean, dtype=np.float32),
        std=np.array(std, dtype=np.float32),
    )
    np.save(year_dir / "idx_train.npy", idx_train.astype(np.int64))
    np.save(year_dir / "idx_val.npy", idx_val.astype(np.int64))
    np.save(year_dir / "idx_test.npy", idx_test.astype(np.int64))


def copy_aux_files(input_root, output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    for filename in AUX_FILES:
        src = input_root / filename
        if src.exists():
            shutil.copy2(src, output_root / filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, required=True)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--steps_per_day", type=int, default=288)
    parser.add_argument("--windows", type=str, default="12:12,24:24")
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--copy_aux", type=int, default=1)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    input_year_dir = input_root / str(args.year)
    ptr = np.load(input_year_dir / "his.npz")

    normalized_full = ptr["data"].astype(np.float32)
    full_mean = float(ptr["mean"])
    full_std = float(ptr["std"])
    raw_full = normalized_full.copy()
    raw_full[..., 0] = raw_full[..., 0] * full_std + full_mean

    start_step, end_step, month_days = month_step_range(args.year, args.month, args.steps_per_day)
    raw_month = raw_full[start_step:end_step].copy()
    if len(raw_month) <= 0:
        raise ValueError("Month slice is empty: start={} end={}".format(start_step, end_step))

    windows = parse_windows(args.windows)
    summary = {
        "input_root": str(input_root),
        "year": int(args.year),
        "month": int(args.month),
        "month_days": int(month_days),
        "steps_per_day": int(args.steps_per_day),
        "start_step": int(start_step),
        "end_step": int(end_step),
        "month_shape": [int(v) for v in raw_month.shape],
        "windows": [],
    }

    for seq_len, horizon in windows:
        output_root = Path("{}_q{}h{}".format(args.output_prefix, seq_len, horizon))
        idx_train, idx_val, idx_test = build_idx(
            num_steps=len(raw_month),
            seq_len=seq_len,
            horizon=horizon,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        month_mean, month_std = compute_train_scaler(raw_month, idx_val, seq_len)
        normalized_month = raw_month.copy()
        normalized_month[..., 0] = (normalized_month[..., 0] - month_mean) / month_std

        save_dataset(
            output_root=output_root,
            year=args.year,
            normalized_data=normalized_month,
            mean=month_mean,
            std=month_std,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
        )
        if int(args.copy_aux):
            copy_aux_files(input_root, output_root)

        window_summary = {
            "output_root": str(output_root),
            "seq_len": int(seq_len),
            "horizon": int(horizon),
            "train_samples": int(len(idx_train)),
            "val_samples": int(len(idx_val)),
            "test_samples": int(len(idx_test)),
            "mean": float(month_mean),
            "std": float(month_std),
        }
        summary["windows"].append(window_summary)

        with open(output_root / "month_subset_summary.json", "w", encoding="utf-8") as f:
            json.dump(window_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
