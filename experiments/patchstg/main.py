import math
import os
import numpy as np
import pandas as pd

import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

import torch

torch.set_num_threads(3)

from src.base.engine import BaseEngine
from src.models.patchstg import PatchSTG
from src.utils.args import get_public_config
from src.utils.dataloader import get_dataset_info
from src.utils.dataloader import load_dataset
from src.utils.dataloader import load_dataset_for_stllm
from src.utils.experiment_naming import build_experiment_dir_name
from src.utils.logging import get_logger
from src.utils.metrics import masked_mae
from src.utils.swanlab_tracker import resolve_swanlab_job_type


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def _read_meta(meta_path):
    meta = pd.read_csv(meta_path, sep="\t", engine="python", on_bad_lines="skip")
    if "station_id" not in meta.columns:
        meta = pd.read_csv(meta_path, low_memory=False)
    meta = meta.copy()
    if "station_id" in meta.columns:
        meta["station_id"] = pd.to_numeric(meta["station_id"], errors="coerce")
        meta = meta.dropna(subset=["station_id"]).copy()
        meta["station_id"] = meta["station_id"].astype(np.int64)
    return meta


def _resolve_node_order_path(args, data_path):
    if str(args.node_order_path).strip():
        return args.node_order_path
    candidates = [
        os.path.join(data_path, "node_order_sacramento.npy"),
        os.path.join(data_path, "node_order.npy"),
        os.path.join(os.path.dirname(data_path), "node_order.npy"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def _resolve_meta_path(args, data_path):
    if str(args.meta_path).strip():
        return args.meta_path
    candidates = [
        os.path.join(data_path, "sensor_meta_feature.csv"),
        os.path.join(os.path.dirname(data_path), "sensor_meta_feature.csv"),
        "/root/XTraffic/data/sensor_meta_feature.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def _build_locations(meta, node_order, node_num):
    if "Lat" not in meta.columns or "Lng" not in meta.columns:
        raise ValueError("meta file must contain Lat and Lng columns")

    if node_order is None:
        ordered_meta = meta.copy()
        ordered_meta["Lat"] = pd.to_numeric(ordered_meta["Lat"], errors="coerce")
        ordered_meta["Lng"] = pd.to_numeric(ordered_meta["Lng"], errors="coerce")
        ordered_meta = ordered_meta.dropna(subset=["Lat", "Lng"]).copy()
        if len(ordered_meta) != int(node_num):
            raise ValueError(
                "meta rows with valid Lat/Lng do not match node_num: {} vs {}".format(
                    len(ordered_meta), node_num
                )
            )
        return np.stack(
            [ordered_meta["Lat"].to_numpy(dtype=np.float32), ordered_meta["Lng"].to_numpy(dtype=np.float32)],
            axis=0,
        )

    if "station_id" not in meta.columns:
        raise ValueError("meta file must contain station_id when node_order_path is used")

    meta = meta.copy()
    meta["Lat"] = pd.to_numeric(meta["Lat"], errors="coerce")
    meta["Lng"] = pd.to_numeric(meta["Lng"], errors="coerce")
    row_map = {int(row["station_id"]): row for _, row in meta.iterrows()}

    lat_list = []
    lng_list = []
    missing = []
    for sid in node_order.astype(np.int64).tolist():
        row = row_map.get(int(sid))
        if row is None or pd.isna(row["Lat"]) or pd.isna(row["Lng"]):
            missing.append(int(sid))
            continue
        lat_list.append(float(row["Lat"]))
        lng_list.append(float(row["Lng"]))

    if missing:
        preview = missing[:10]
        raise ValueError(
            "meta_path is missing valid Lat/Lng for {} node(s), examples: {}".format(
                len(missing), preview
            )
        )

    return np.stack(
        [np.asarray(lat_list, dtype=np.float32), np.asarray(lng_list, dtype=np.float32)],
        axis=0,
    )


def _get_cache_root():
    data_root = os.getenv("LARGEST_DATA_ROOT", "/public_data/LargeST_data")
    cache_root = os.path.join(data_root, "cache", "patchstg")
    os.makedirs(cache_root, exist_ok=True)
    return cache_root


def _resolve_cache_paths(args, data_path, node_num, recur_times, spa_patchsize):
    data_token = os.path.basename(os.path.normpath(data_path))
    common = "{}_{}_n{}_q{}_r{}_sps{}".format(
        data_token,
        args.years,
        int(node_num),
        int(args.seq_len),
        int(recur_times),
        int(spa_patchsize),
    )
    cache_root = _get_cache_root()

    sim_cache_path = str(args.sim_cache_path).strip() or os.path.join(
        cache_root, "sim_{}.npy".format(common)
    )
    spatial_index_cache_path = str(args.spatial_index_cache_path).strip() or os.path.join(
        cache_root, "indices_{}.npz".format(common)
    )
    return sim_cache_path, spatial_index_cache_path


def _construct_similarity_adj(train_data, steps_per_day):
    data = np.asarray(train_data, dtype=np.float32)
    if data.ndim != 3 or data.shape[-1] != 1:
        raise ValueError("train_data must be [T, N, 1], got {}".format(data.shape))

    period = max(1, int(steps_per_day))
    num_periods = data.shape[0] // period
    if num_periods > 0:
        usable = num_periods * period
        mean_pattern = data[:usable].reshape(num_periods, period, data.shape[1], 1).mean(axis=0)
    else:
        mean_pattern = data

    node_series = mean_pattern.squeeze(-1).transpose(1, 0)
    norms = np.linalg.norm(node_series, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-6, a_max=None)
    sim = (node_series @ node_series.T) / (norms @ norms.T)
    sim_mean = float(sim.mean())
    sim_std = float(sim.std())
    if sim_std < 1e-6:
        sim_std = 1.0
    sim = np.exp((sim - sim_mean) / sim_std)
    return sim.astype(np.float32)


def _augment_align(sim_matrix, aug_len):
    if aug_len <= 0:
        return np.empty((0,), dtype=np.int64)
    flat_sorted = np.argsort(sim_matrix.reshape(-1) * -1)
    sorted_idx = flat_sorted % sim_matrix.shape[-1]
    aug_idx = []
    for idx in sorted_idx.tolist():
        if idx not in aug_idx:
            aug_idx.append(int(idx))
        if len(aug_idx) == aug_len:
            break
    return np.asarray(aug_idx, dtype=np.int64)


def _reorder_data(parts_idx, adj, spa_patchsize):
    ori_parts_idx = np.array([], dtype=np.int64)
    reo_parts_idx = np.array([], dtype=np.int64)
    reo_all_idx = np.array([], dtype=np.int64)
    for i, part_idx in enumerate(parts_idx):
        part_dist = adj[part_idx, :].copy()
        part_dist[:, part_idx] = 0
        if spa_patchsize - part_idx.shape[0] > 0:
            local_part_idx = _augment_align(part_dist, spa_patchsize - part_idx.shape[0])
            auged_part_idx = np.concatenate([part_idx, local_part_idx], axis=0)
        else:
            auged_part_idx = part_idx

        reo_parts_idx = np.concatenate(
            [reo_parts_idx, np.arange(part_idx.shape[0], dtype=np.int64) + spa_patchsize * i],
            axis=0,
        )
        ori_parts_idx = np.concatenate([ori_parts_idx, part_idx.astype(np.int64)], axis=0)
        reo_all_idx = np.concatenate([reo_all_idx, auged_part_idx.astype(np.int64)], axis=0)
    return ori_parts_idx, reo_parts_idx, reo_all_idx


def _kd_tree(locations, times, axis):
    sorted_idx = np.argsort(locations[axis])
    midpoint = locations.shape[1] // 2
    part1 = np.sort(sorted_idx[:midpoint])
    part2 = np.sort(sorted_idx[midpoint:])
    if times == 1:
        return [part1, part2]

    parts = []
    left_parts = _kd_tree(locations[:, part1], times - 1, axis ^ 1)
    right_parts = _kd_tree(locations[:, part2], times - 1, axis ^ 1)
    for part in left_parts:
        parts.append(part1[part])
    for part in right_parts:
        parts.append(part2[part])
    return parts


def _select_factor(spa_patchnum, target_factor):
    factor = max(1, min(int(target_factor), int(spa_patchnum)))
    while factor > 1 and spa_patchnum % factor != 0:
        factor //= 2
    return max(1, factor)


def _resolve_patch_params(args, node_num, logger):
    tem_patchsize = int(args.tem_patchsize)
    if tem_patchsize <= 0:
        tem_patchsize = min(12, int(args.seq_len))

    tem_patchnum = int(args.tem_patchnum)
    if tem_patchnum <= 0:
        if args.seq_len % tem_patchsize == 0:
            tem_patchnum = args.seq_len // tem_patchsize
        else:
            tem_patchsize = int(args.seq_len)
            tem_patchnum = 1

    if tem_patchsize * tem_patchnum != int(args.seq_len):
        raise ValueError(
            "PatchSTG requires tem_patchsize * tem_patchnum == seq_len, got {} * {} != {}".format(
                tem_patchsize, tem_patchnum, args.seq_len
            )
        )

    spa_patchsize = max(1, int(args.spa_patchsize))
    max_recur = max(1, int(math.floor(math.log2(max(2, int(node_num))))))
    recur_times = int(args.recur_times)
    if recur_times <= 0:
        recur_times = int(math.ceil(math.log2(max(2, int(math.ceil(float(node_num) / spa_patchsize))))))
    recur_times = min(recur_times, max_recur)
    spa_patchnum = 2 ** recur_times
    factors = _select_factor(spa_patchnum, args.factors)

    logger.info(
        "PatchSTG patch config: tps={}, tpn={}, recur={}, sps={}, spn={}, factors={}".format(
            tem_patchsize, tem_patchnum, recur_times, spa_patchsize, spa_patchnum, factors
        )
    )
    return tem_patchsize, tem_patchnum, recur_times, spa_patchsize, spa_patchnum, factors


def _load_train_history(data_path, args, traffic_dim):
    ptr = np.load(os.path.join(data_path, args.years, "his.npz"))
    idx_train = np.load(os.path.join(data_path, args.years, "idx_train.npy"))
    if len(idx_train) == 0:
        raise ValueError("idx_train.npy is empty")

    max_anchor = int(np.max(idx_train))
    train_end = max_anchor + 1
    train_history = ptr["data"][:train_end, :, :traffic_dim]
    return train_history


def _load_spatial_indices(
    args,
    data_path,
    node_num,
    meta_path,
    node_order_path,
    recur_times,
    spa_patchsize,
    steps_per_day,
    logger,
):
    sim_cache_path, spatial_index_cache_path = _resolve_cache_paths(
        args, data_path, node_num, recur_times, spa_patchsize
    )
    logger.info("PatchSTG sim cache path: " + sim_cache_path)
    logger.info("PatchSTG spatial index cache path: " + spatial_index_cache_path)

    if os.path.exists(spatial_index_cache_path):
        cache = np.load(spatial_index_cache_path)
        ori_parts_idx = cache["ori_parts_idx"].astype(np.int64)
        reo_parts_idx = cache["reo_parts_idx"].astype(np.int64)
        reo_all_idx = cache["reo_all_idx"].astype(np.int64)
        if len(ori_parts_idx) == int(node_num):
            return ori_parts_idx, reo_parts_idx, reo_all_idx
        logger.info("PatchSTG spatial cache invalidated due to node_num mismatch.")

    meta = _read_meta(meta_path)
    node_order = np.load(node_order_path) if node_order_path else None
    locations = _build_locations(meta, node_order, node_num)

    if os.path.exists(sim_cache_path):
        adj = np.load(sim_cache_path).astype(np.float32)
    else:
        train_history = _load_train_history(data_path, args, traffic_dim=1)
        adj = _construct_similarity_adj(train_history, steps_per_day=steps_per_day)
        np.save(sim_cache_path, adj)

    parts_idx = _kd_tree(locations, recur_times, 0)
    ori_parts_idx, reo_parts_idx, reo_all_idx = _reorder_data(parts_idx, adj, spa_patchsize)
    np.savez(
        spatial_index_cache_path,
        ori_parts_idx=ori_parts_idx,
        reo_parts_idx=reo_parts_idx,
        reo_all_idx=reo_all_idx,
    )
    logger.info("PatchSTG padded nodes: {}".format(len(reo_all_idx)))
    return ori_parts_idx, reo_parts_idx, reo_all_idx


def get_config():
    parser = get_public_config()
    parser.set_defaults(bs=16, max_epochs=50, patience=50, input_dim=1)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--node_num", type=int, default=0)
    parser.add_argument("--meta_path", type=str, default="")
    parser.add_argument("--node_order_path", type=str, default="")
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--experiment_timestamp", type=str, default="")

    parser.add_argument("--traffic_dim", type=int, default=1)
    parser.add_argument("--steps_per_day", type=int, default=288)
    parser.add_argument("--day_of_week_size", type=int, default=7)
    parser.add_argument("--auto_time_features", type=int, default=1)
    parser.add_argument("--add_time_of_day", type=int, default=1)
    parser.add_argument("--add_day_of_week", type=int, default=1)
    parser.add_argument("--time_start_offset", type=int, default=0)
    parser.add_argument("--time_day_idx", type=int, default=1)
    parser.add_argument("--day_in_week_idx", type=int, default=2)

    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--tem_patchsize", type=int, default=12)
    parser.add_argument("--tem_patchnum", type=int, default=0)
    parser.add_argument("--recur_times", type=int, default=0)
    parser.add_argument("--spa_patchsize", type=int, default=2)
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--input_emb_dims", type=int, default=64)
    parser.add_argument("--node_dims", type=int, default=64)
    parser.add_argument("--tod_dims", type=int, default=32)
    parser.add_argument("--dow_dims", type=int, default=32)

    parser.add_argument("--lrate", type=float, default=2e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--milestones", type=str, default="1,35,40")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--clip_grad_value", type=float, default=5.0)

    parser.add_argument("--sim_cache_path", type=str, default="")
    parser.add_argument("--spatial_index_cache_path", type=str, default="")

    parser.add_argument("--use_swanlab", type=int, default=1)
    parser.add_argument("--swanlab_project", type=str, default="LargeST")
    parser.add_argument("--swanlab_experiment", type=str, default="")
    parser.add_argument("--swanlab_mode", type=str, default="cloud")
    parser.add_argument("--swanlab_description", type=str, default="")
    parser.add_argument("--desc", dest="swanlab_description", type=str)
    parser.add_argument(
        "--swanlab_lark_webhook_url",
        type=str,
        default=os.getenv("SWANLAB_LARK_WEBHOOK_URL", ""),
    )
    parser.add_argument(
        "--swanlab_lark_secret",
        type=str,
        default=os.getenv("SWANLAB_LARK_SECRET", ""),
    )
    args = parser.parse_args()

    if int(args.traffic_dim) != 1:
        raise ValueError("PatchSTG currently supports flow-only forecasting, so traffic_dim must be 1")

    folder_name = build_experiment_dir_name(
        model_name=args.model_name,
        dataset=args.dataset,
        years=args.years,
        seq_len=args.seq_len,
        horizon=args.horizon,
        seed=args.seed,
        extra_parts=[
            ("flow", args.traffic_dim),
            ("layers", args.layers),
        ],
        run_tag=args.run_tag,
        started_at=str(args.experiment_timestamp).strip() or None,
    )
    log_dir = "./experiments/{}/{}/".format(args.model_name, folder_name)
    logger = get_logger(log_dir, __name__, "record_s{}.log".format(args.seed))
    logger.info(args)
    return args, log_dir, logger, folder_name


def resolve_data_info(args, logger):
    if args.data_path:
        data_path = args.data_path
        if args.node_num > 0:
            node_num = args.node_num
        else:
            ptr = np.load(os.path.join(data_path, args.years, "his.npz"))
            node_num = int(ptr["data"].shape[1])
        meta_path = _resolve_meta_path(args, data_path)
        if not meta_path:
            raise ValueError(
                "PatchSTG requires meta_path with Lat/Lng information. "
                "Please pass --meta_path explicitly."
            )
        node_order_path = _resolve_node_order_path(args, data_path)
        logger.info("Use custom dataset path.")
        return data_path, node_num, meta_path, node_order_path

    data_path, _, node_num = get_dataset_info(args.dataset)
    meta_path = _resolve_meta_path(args, data_path)
    if not meta_path:
        raise ValueError(
            "PatchSTG requires meta_path with Lat/Lng information. Please pass --meta_path."
        )
    node_order_path = _resolve_node_order_path(args, data_path)
    return data_path, node_num, meta_path, node_order_path


def _parse_milestones(text):
    values = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    return values


def main():
    args, log_dir, logger, folder_name = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    data_path, node_num, meta_path, node_order_path = resolve_data_info(args, logger)
    logger.info("PatchSTG meta path: " + meta_path)
    if node_order_path:
        logger.info("PatchSTG node order path: " + node_order_path)

    tem_patchsize, tem_patchnum, recur_times, spa_patchsize, spa_patchnum, factors = _resolve_patch_params(
        args, node_num, logger
    )

    if bool(args.auto_time_features):
        base_input_dim = int(args.traffic_dim)
        add_time_of_day = bool(args.add_time_of_day)
        add_day_of_week = bool(args.add_day_of_week)
        model_input_dim = base_input_dim + int(add_time_of_day) + int(add_day_of_week)
        dataloader, scaler = load_dataset_for_stllm(
            data_path=data_path,
            args=args,
            logger=logger,
            base_input_dim=base_input_dim,
            steps_per_day=args.steps_per_day,
            add_time_of_day=add_time_of_day,
            add_day_of_week=add_day_of_week,
            time_start_offset=args.time_start_offset,
        )
        time_day_idx = base_input_dim if add_time_of_day else -1
        day_in_week_idx = base_input_dim + int(add_time_of_day) if add_day_of_week else -1
    else:
        model_input_dim = args.input_dim
        dataloader, scaler = load_dataset(data_path, args, logger)
        time_day_idx = args.time_day_idx
        day_in_week_idx = args.day_in_week_idx

    ori_parts_idx, reo_parts_idx, reo_all_idx = _load_spatial_indices(
        args=args,
        data_path=data_path,
        node_num=node_num,
        meta_path=meta_path,
        node_order_path=node_order_path,
        recur_times=recur_times,
        spa_patchsize=spa_patchsize,
        steps_per_day=args.steps_per_day,
        logger=logger,
    )

    model = PatchSTG(
        node_num=node_num,
        input_dim=model_input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        ori_parts_idx=ori_parts_idx,
        reo_parts_idx=reo_parts_idx,
        reo_all_idx=reo_all_idx,
        tem_patchsize=tem_patchsize,
        tem_patchnum=tem_patchnum,
        spa_patchsize=spa_patchsize,
        spa_patchnum=spa_patchnum,
        factors=factors,
        layers=args.layers,
        steps_per_day=args.steps_per_day,
        day_of_week_size=args.day_of_week_size,
        input_emb_dims=args.input_emb_dims,
        node_dims=args.node_dims,
        tod_dims=args.tod_dims,
        dow_dims=args.dow_dims,
        traffic_dim=args.traffic_dim,
        time_day_idx=time_day_idx,
        day_in_week_idx=day_in_week_idx,
        dropout=args.dropout,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    milestones = _parse_milestones(args.milestones)
    scheduler = None
    if milestones:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=args.gamma,
        )

    engine = BaseEngine(
        device=device,
        model=model,
        dataloader=dataloader,
        scaler=scaler,
        sampler=None,
        loss_fn=masked_mae,
        lrate=args.lrate,
        optimizer=optimizer,
        scheduler=scheduler,
        clip_grad_value=args.clip_grad_value,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_dir=log_dir,
        logger=logger,
        seed=args.seed,
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
