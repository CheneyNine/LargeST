#!/usr/bin/env python3
import argparse
import json
import os
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
    "county_candidates.json",
    "match_incidents_y2023_sacramento.csv",
    "summary.json",
]


def filter_idx_by_incident(idx, incident_flags, seq_len, horizon, criterion):
    keep_event = []
    keep_noevent = []

    for center in idx.astype(np.int64).tolist():
        if criterion == "forecast":
            start = center + 1
            end = center + horizon + 1
        elif criterion == "history_forecast":
            start = center - seq_len + 1
            end = center + horizon + 1
        else:
            raise ValueError("Unsupported criterion: {}".format(criterion))

        has_incident = bool(incident_flags[:, start:end].any())
        if has_incident:
            keep_event.append(center)
        else:
            keep_noevent.append(center)

    return np.asarray(keep_event, dtype=np.int64), np.asarray(keep_noevent, dtype=np.int64)


def ensure_symlink(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def populate_output_dir(src_root, out_root, year):
    out_root.mkdir(parents=True, exist_ok=True)
    for name in AUX_FILES:
        src = src_root / name
        if not src.exists():
            continue
        dst = out_root / name
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if src.is_file():
            if src.stat().st_size > 64 * 1024 * 1024:
                dst.symlink_to(src)
            else:
                shutil.copy2(src, dst)

    year_src = src_root / str(year) / "his.npz"
    year_dst = out_root / str(year) / "his.npz"
    ensure_symlink(year_src, year_dst)

    ntf_src = src_root / "traffic_incident_ntf.npy"
    if ntf_src.exists():
        ensure_symlink(ntf_src, out_root / "traffic_incident_ntf.npy")


def save_idx_only_dataset(src_root, out_root, year, idx_map, summary):
    populate_output_dir(src_root, out_root, year)
    year_dir = out_root / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    for cat, idx in idx_map.items():
        np.save(year_dir / "idx_{}.npy".format(cat), idx.astype(np.int64))

    with open(out_root / "incident_window_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--criterion", type=str, default="forecast")
    parser.add_argument("--incident_tensor_path", type=str, default="")
    parser.add_argument("--incident_channel", type=int, default=3)
    parser.add_argument("--event_output_root", type=str, default="")
    parser.add_argument("--noevent_output_root", type=str, default="")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    year_dir = input_root / str(args.year)
    if not year_dir.exists():
        raise FileNotFoundError("Missing year dir: {}".format(year_dir))

    incident_tensor_path = (
        Path(args.incident_tensor_path).resolve()
        if args.incident_tensor_path
        else (input_root / "traffic_incident_ntf.npy")
    )
    if not incident_tensor_path.exists():
        raise FileNotFoundError("Missing incident tensor: {}".format(incident_tensor_path))

    event_output_root = (
        Path(args.event_output_root).resolve()
        if args.event_output_root
        else input_root.parent / "{}_event_q{}h{}".format(input_root.name, args.seq_len, args.horizon)
    )
    noevent_output_root = (
        Path(args.noevent_output_root).resolve()
        if args.noevent_output_root
        else input_root.parent / "{}_noevent_q{}h{}".format(input_root.name, args.seq_len, args.horizon)
    )

    his_ptr = np.load(year_dir / "his.npz")
    num_steps = int(his_ptr["data"].shape[0])
    num_nodes = int(his_ptr["data"].shape[1])
    incident_ntf = np.load(incident_tensor_path, mmap_mode="r")
    if incident_ntf.ndim != 3:
        raise ValueError("incident tensor must be 3D, got {}".format(incident_ntf.shape))
    if int(incident_ntf.shape[0]) != num_nodes or int(incident_ntf.shape[1]) != num_steps:
        raise ValueError(
            "incident tensor shape mismatch: {} vs expected ({}, {}, F)".format(
                incident_ntf.shape, num_nodes, num_steps
            )
        )

    incident_flags = incident_ntf[:, :, int(args.incident_channel)] != 0

    event_idx_map = {}
    noevent_idx_map = {}
    split_summary = {
        "input_root": str(input_root),
        "year": int(args.year),
        "seq_len": int(args.seq_len),
        "horizon": int(args.horizon),
        "criterion": str(args.criterion),
        "incident_tensor_path": str(incident_tensor_path),
        "incident_channel": int(args.incident_channel),
        "num_steps": int(num_steps),
        "num_nodes": int(num_nodes),
        "splits": {},
    }

    for cat in ["train", "val", "test"]:
        idx = np.load(year_dir / "idx_{}.npy".format(cat))
        event_idx, noevent_idx = filter_idx_by_incident(
            idx=idx,
            incident_flags=incident_flags,
            seq_len=int(args.seq_len),
            horizon=int(args.horizon),
            criterion=str(args.criterion),
        )
        event_idx_map[cat] = event_idx
        noevent_idx_map[cat] = noevent_idx
        split_summary["splits"][cat] = {
            "original": int(len(idx)),
            "event": int(len(event_idx)),
            "noevent": int(len(noevent_idx)),
        }

    split_summary["event_output_root"] = str(event_output_root)
    split_summary["noevent_output_root"] = str(noevent_output_root)

    save_idx_only_dataset(
        src_root=input_root,
        out_root=event_output_root,
        year=args.year,
        idx_map=event_idx_map,
        summary={**split_summary, "subset": "event"},
    )
    save_idx_only_dataset(
        src_root=input_root,
        out_root=noevent_output_root,
        year=args.year,
        idx_map=noevent_idx_map,
        summary={**split_summary, "subset": "noevent"},
    )

    print(json.dumps(split_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
