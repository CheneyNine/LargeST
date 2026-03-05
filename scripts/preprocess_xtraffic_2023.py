#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_counties(counties_arg):
    if not counties_arg:
        return None
    counties = [item.strip() for item in counties_arg.split(",") if item.strip()]
    return counties if counties else None


def read_and_clean_meta(meta_path):
    meta = pd.read_csv(meta_path, sep="\t", engine="python", on_bad_lines="skip", low_memory=False)

    for col in ["City", "Sensor Type"]:
        if col in meta.columns:
            meta = meta.drop(columns=[col])

    if "Length" in meta.columns:
        meta["Length"] = pd.to_numeric(meta["Length"], errors="coerce").fillna(0.0)

    for col in ["station_id", "Lat", "Lng", "Fwy", "Abs PM"]:
        if col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce")

    required_cols = ["station_id", "Lat", "Lng"]
    missing_cols = [col for col in required_cols if col not in meta.columns]
    if missing_cols:
        raise ValueError("Meta file missing required columns: {}".format(missing_cols))

    meta = meta.dropna(subset=["station_id", "Lat", "Lng"]).copy()
    meta["station_id"] = meta["station_id"].astype(np.int64)
    return meta


def get_month_files(year_dir):
    files = sorted(year_dir.glob("p*_done.npy"))
    if len(files) != 12:
        raise ValueError("Expect 12 monthly files under {}, got {}".format(year_dir, len(files)))
    return files


def scan_traffic_stats(files, node_num):
    nan_count = np.zeros(node_num, dtype=np.int64)
    total_count = np.zeros(node_num, dtype=np.int64)
    month_lengths = []

    for f in files:
        arr = np.load(f, mmap_mode="r")
        if arr.ndim != 3:
            raise ValueError("Traffic file {} must be 3D, got {}".format(f, arr.shape))
        if arr.shape[1] != node_num:
            raise ValueError("Node count mismatch in {}: {} vs {}".format(f, arr.shape[1], node_num))
        if arr.shape[2] < 3:
            raise ValueError("Expect at least 3 channels in {}, got {}".format(f, arr.shape[2]))

        month_lengths.append(int(arr.shape[0]))
        arr3 = arr[:, :, :3]
        nan_count += np.isnan(arr3).sum(axis=(0, 2))
        total_count += arr3.shape[0] * arr3.shape[2]

    missing_ratio = nan_count / np.maximum(total_count, 1)
    return missing_ratio, month_lengths, int(sum(month_lengths))


def write_filtered_traffic(files, selected_node_idx, total_steps, output_path, node_chunk):
    n_sel = len(selected_node_idx)
    out = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float64, shape=(n_sel, total_steps, 5))
    out[:, :, 3:] = 0.0

    cursor = 0
    for f in files:
        arr = np.load(f, mmap_mode="r")
        t_len = arr.shape[0]

        for start in range(0, n_sel, node_chunk):
            end = min(n_sel, start + node_chunk)
            chunk_nodes = selected_node_idx[start:end]
            chunk = arr[:, chunk_nodes, :3]
            out[start:end, cursor : cursor + t_len, :3] = np.transpose(chunk, (1, 0, 2))

        cursor += t_len

    return out


def resolve_incident_node_idx(incidents, node_order):
    if "node_idx" in incidents.columns:
        incidents["node_idx"] = pd.to_numeric(incidents["node_idx"], errors="coerce")
    elif "station_id" in incidents.columns:
        station_to_idx = {int(station_id): int(idx) for idx, station_id in enumerate(node_order.tolist())}
        station = pd.to_numeric(incidents["station_id"], errors="coerce")
        incidents["node_idx"] = station.map(
            lambda x: station_to_idx.get(int(x), np.nan) if pd.notna(x) else np.nan
        )
    else:
        raise ValueError("Incident file must contain node_idx or station_id")

    incidents["node_idx"] = incidents["node_idx"].astype("Int64")
    return incidents


def align_incidents_to_tensor(
    out_tensor,
    incidents,
    selected_node_idx,
    year,
    freq_minutes,
):
    old_to_new = {int(old_idx): int(new_idx) for new_idx, old_idx in enumerate(selected_node_idx.tolist())}
    total_steps = out_tensor.shape[1]
    start_time = pd.Timestamp("{}-01-01 00:00:00".format(year))

    incidents = incidents.copy()
    incidents = incidents.dropna(subset=["dt"]).copy()
    incidents["dt"] = pd.to_datetime(incidents["dt"], errors="coerce")
    incidents = incidents.dropna(subset=["dt"]).copy()

    incidents["incident_id"] = pd.to_numeric(incidents["incident_id"], errors="coerce").fillna(0).astype(np.int64)
    incidents["duration"] = pd.to_numeric(incidents.get("duration", 0), errors="coerce").fillna(0.0).clip(lower=0.0)
    incidents = incidents.dropna(subset=["node_idx"]).copy()
    incidents["node_idx"] = incidents["node_idx"].astype(int)
    incidents = incidents[incidents["node_idx"].isin(old_to_new)].copy()

    start_idx = ((incidents["dt"] - start_time).dt.total_seconds() // (freq_minutes * 60)).astype(np.int64)
    step_len = np.maximum(1, np.ceil(incidents["duration"].to_numpy() / freq_minutes).astype(np.int64))
    node_idx = incidents["node_idx"].to_numpy()
    incident_ids = incidents["incident_id"].to_numpy()

    assignments = 0
    collisions = 0
    dropped_out_of_range = 0

    for i in range(len(incidents)):
        s = int(start_idx.iloc[i]) if hasattr(start_idx, "iloc") else int(start_idx[i])
        if s >= total_steps:
            dropped_out_of_range += 1
            continue

        e = min(total_steps, s + int(step_len[i]))
        if e <= 0:
            dropped_out_of_range += 1
            continue
        s = max(s, 0)
        if e <= s:
            dropped_out_of_range += 1
            continue

        new_node = old_to_new[int(node_idx[i])]
        event_steps = int(step_len[i])
        prog = np.arange(1, e - s + 1, dtype=np.float64) / float(event_steps)

        segment_id = out_tensor[new_node, s:e, 3]
        segment_prog = out_tensor[new_node, s:e, 4]
        replace_mask = (segment_id == 0) | (prog > segment_prog)

        collisions += int((~(segment_id == 0) & replace_mask).sum())
        assignments += int(replace_mask.sum())

        segment_id[replace_mask] = float(incident_ids[i])
        segment_prog[replace_mask] = prog[replace_mask]

        out_tensor[new_node, s:e, 3] = segment_id
        out_tensor[new_node, s:e, 4] = segment_prog

    return {
        "incident_rows_after_clean": int(len(incidents)),
        "assigned_slots": int(assignments),
        "collision_overwrites": int(collisions),
        "dropped_out_of_range": int(dropped_out_of_range),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/root/XTraffic/data")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--missing_ratio_threshold", type=float, default=0.2)
    parser.add_argument("--counties", type=str, default="")
    parser.add_argument("--incident_match_path", type=str, default="")
    parser.add_argument("--time_freq_minutes", type=int, default=5)
    parser.add_argument("--node_chunk", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    base_dir = Path(args.data_dir).resolve()
    year_dir = base_dir / "year_{}".format(args.year)
    if not year_dir.exists():
        raise FileNotFoundError("Year directory not found: {}".format(year_dir))

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (base_dir / "processed_y{}".format(args.year))
    output_dir.mkdir(parents=True, exist_ok=True)

    incident_match_path = (
        Path(args.incident_match_path).resolve()
        if args.incident_match_path
        else (base_dir / "match_incidents_y{}_nearest_with_node_idx.csv".format(args.year))
    )

    node_order = np.load(base_dir / "node_order.npy")
    month_files = get_month_files(year_dir)

    meta = read_and_clean_meta(base_dir / "sensor_meta_feature.csv")
    county_candidates = sorted(meta["County"].dropna().astype(str).unique().tolist()) if "County" in meta.columns else []
    selected_counties = parse_counties(args.counties)
    if selected_counties:
        if "County" not in meta.columns:
            raise ValueError("Meta file does not contain County column")
        missing_counties = [county for county in selected_counties if county not in county_candidates]
        if missing_counties:
            raise ValueError("Unknown county options: {}. Available: {}".format(missing_counties, county_candidates))
        meta = meta[meta["County"].isin(selected_counties)].copy()

    station_to_node = {int(station_id): int(idx) for idx, station_id in enumerate(node_order.tolist())}
    meta["node_idx"] = meta["station_id"].map(
        lambda x: station_to_node.get(int(x), np.nan) if pd.notna(x) else np.nan
    )
    meta = meta.dropna(subset=["node_idx"]).copy()
    meta["node_idx"] = meta["node_idx"].astype(int)

    missing_ratio, month_lengths, total_steps = scan_traffic_stats(month_files, len(node_order))
    mask_meta = np.zeros(len(node_order), dtype=bool)
    mask_meta[meta["node_idx"].to_numpy()] = True
    mask_missing = missing_ratio <= args.missing_ratio_threshold
    selected_mask = mask_meta & mask_missing
    selected_node_idx = np.where(selected_mask)[0]

    if len(selected_node_idx) == 0:
        raise ValueError("No nodes left after filtering. Please relax filters.")

    selected_station_id = node_order[selected_node_idx]
    selected_meta = meta[meta["node_idx"].isin(selected_node_idx)].copy().sort_values("node_idx")
    selected_meta["missing_ratio"] = selected_meta["node_idx"].map(lambda x: float(missing_ratio[int(x)]))

    np.save(output_dir / "selected_node_idx.npy", selected_node_idx)
    np.save(output_dir / "selected_station_id.npy", selected_station_id)
    selected_meta.to_csv(output_dir / "sensor_meta_feature_selected.csv", index=False)

    tensor_path = output_dir / "traffic_incident_ntf.npy"
    out_tensor = write_filtered_traffic(
        month_files,
        selected_node_idx,
        total_steps,
        tensor_path,
        node_chunk=args.node_chunk,
    )

    incidents = pd.read_csv(incident_match_path, low_memory=False)
    incidents = resolve_incident_node_idx(incidents, node_order)
    incident_stats = align_incidents_to_tensor(
        out_tensor,
        incidents,
        selected_node_idx,
        year=args.year,
        freq_minutes=args.time_freq_minutes,
    )

    summary = {
        "year": int(args.year),
        "month_files": [f.name for f in month_files],
        "month_lengths": [int(v) for v in month_lengths],
        "total_steps": int(total_steps),
        "original_node_num": int(len(node_order)),
        "meta_filtered_node_num": int(mask_meta.sum()),
        "missing_filtered_node_num": int(mask_missing.sum()),
        "selected_node_num": int(len(selected_node_idx)),
        "missing_ratio_threshold": float(args.missing_ratio_threshold),
        "county_candidates": county_candidates,
        "selected_counties": selected_counties if selected_counties else county_candidates,
        "incident_match_path": str(incident_match_path),
        "output_tensor": str(tensor_path),
        "output_shape_ntf": [int(v) for v in out_tensor.shape],
    }
    summary.update(incident_stats)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(output_dir / "county_candidates.json", "w", encoding="utf-8") as f:
        json.dump(county_candidates, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
