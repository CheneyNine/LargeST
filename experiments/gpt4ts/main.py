import os
import numpy as np

import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

import torch

torch.set_num_threads(3)

from src.base.engine import BaseEngine
from src.models.gpt4ts import GPT4TS
from src.utils.args import get_public_config
from src.utils.dataloader import get_dataset_info
from src.utils.dataloader import load_dataset
from src.utils.experiment_naming import build_experiment_dir_name
from src.utils.logging import get_logger
from src.utils.metrics import masked_mse
from src.utils.swanlab_tracker import resolve_swanlab_job_type


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.set_defaults(bs=16, max_epochs=10, patience=3)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--node_num", type=int, default=0)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--experiment_timestamp", type=str, default="")

    parser.add_argument("--traffic_dim", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--gpt_layers", type=int, default=6)
    parser.add_argument("--is_gpt", type=int, default=1)
    parser.add_argument("--pretrain", type=int, default=1)
    parser.add_argument("--freeze", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gpt_model_name", type=str, default="gpt2")
    parser.add_argument("--gpt_local_files_only", type=int, default=1)
    parser.add_argument("--gpt_allow_download", type=int, default=1)

    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
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

    folder_name = build_experiment_dir_name(
        model_name=args.model_name,
        dataset=args.dataset,
        years=args.years,
        seq_len=args.seq_len,
        horizon=args.horizon,
        seed=args.seed,
        extra_parts=[
            ("dm", args.d_model),
            ("gpt", args.gpt_layers),
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
        logger.info("Use custom dataset path.")
        return data_path, node_num

    data_path, _, node_num = get_dataset_info(args.dataset)
    return data_path, node_num


def main():
    args, log_dir, logger, folder_name = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    data_path, node_num = resolve_data_info(args, logger)
    dataloader, scaler = load_dataset(data_path, args, logger)

    model = GPT4TS(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        traffic_dim=args.traffic_dim,
        d_model=args.d_model,
        patch_size=args.patch_size,
        stride=args.stride,
        gpt_layers=args.gpt_layers,
        is_gpt=args.is_gpt,
        pretrain=args.pretrain,
        freeze=args.freeze,
        dropout=args.dropout,
        gpt_model_name=args.gpt_model_name,
        gpt_local_files_only=args.gpt_local_files_only,
        gpt_allow_download=args.gpt_allow_download,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=min(int(args.max_epochs), 10), eta_min=1e-8
    )

    engine = BaseEngine(
        device=device,
        model=model,
        dataloader=dataloader,
        scaler=scaler,
        sampler=None,
        loss_fn=masked_mse,
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
