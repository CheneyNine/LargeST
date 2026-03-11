import os
import numpy as np

import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

import torch

torch.set_num_threads(3)

from src.engines.steve_engine import STEVE_Engine
from src.models.steve import STEVE
from src.utils.args import get_public_config
from src.utils.dataloader import get_dataset_info
from src.utils.dataloader import load_adj_from_numpy
from src.utils.dataloader import load_dataset
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


def get_config():
    parser = get_public_config()
    parser.add_argument("--adj_type", type=str, default="doubletransition")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--adj_path", type=str, default="")
    parser.add_argument("--node_num", type=int, default=0)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--experiment_timestamp", type=str, default="")

    parser.add_argument("--traffic_dim", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--ks", type=int, default=3)
    parser.add_argument("--kt", type=int, default=3)
    parser.add_argument("--st_dropout", type=float, default=0.1)

    parser.add_argument("--bank_ratio", type=float, default=4.0)
    parser.add_argument("--bank_gamma", type=float, default=0.9)
    parser.add_argument("--temporal_classes", type=int, default=8)
    parser.add_argument("--congestion_channel", type=int, default=1)
    parser.add_argument("--spatial_sample_size", type=int, default=512)
    parser.add_argument("--mi_sample_size", type=int, default=4096)
    parser.add_argument("--grl_alpha", type=float, default=0.01)
    parser.add_argument("--use_grl", type=int, default=1)

    parser.add_argument("--variant_loss_weight", type=float, default=0.3)
    parser.add_argument("--invariant_loss_weight", type=float, default=0.3)
    parser.add_argument("--mi_loss_weight", type=float, default=0.05)

    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.95)
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

    if args.traffic_dim > args.input_dim:
        raise ValueError(
            "traffic_dim must be <= input_dim, got {} > {}".format(args.traffic_dim, args.input_dim)
        )

    folder_name = build_experiment_dir_name(
        model_name=args.model_name,
        dataset=args.dataset,
        years=args.years,
        seq_len=args.seq_len,
        horizon=args.horizon,
        seed=args.seed,
        extra_parts=[
            ("flow", args.traffic_dim),
            ("emb", args.embed_dim),
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
    norm_adj_mx = normalize_adj_mx(adj_mx, args.adj_type)

    base_adj = norm_adj_mx[0]
    base_adj = 0.5 * (base_adj + base_adj.T)
    base_adj = np.maximum(base_adj, 0.0)
    base_adj = torch.tensor(base_adj, dtype=torch.float32, device=device)

    dataloader, scaler = load_dataset(data_path, args, logger)

    model = STEVE(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        base_adj=base_adj,
        traffic_dim=args.traffic_dim,
        embed_dim=args.embed_dim,
        ks=args.ks,
        kt=args.kt,
        st_dropout=args.st_dropout,
        bank_ratio=args.bank_ratio,
        bank_gamma=args.bank_gamma,
        temporal_classes=args.temporal_classes,
        congestion_channel=args.congestion_channel,
        spatial_sample_size=args.spatial_sample_size,
        mi_sample_size=args.mi_sample_size,
        grl_alpha=args.grl_alpha,
        use_grl=bool(args.use_grl),
    )

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    engine = STEVE_Engine(
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
        variant_loss_weight=args.variant_loss_weight,
        invariant_loss_weight=args.invariant_loss_weight,
        mi_loss_weight=args.mi_loss_weight,
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
