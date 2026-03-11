import glob
import os
import numpy as np

import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

import torch

torch.set_num_threads(3)

from src.base.engine import BaseEngine
from src.models.stllm_plus import STLLMPlus
from src.utils.args import get_public_config
from src.utils.dataloader import get_dataset_info
from src.utils.dataloader import load_adj_from_numpy
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


def infer_adj_path(data_path):
    candidates = sorted(glob.glob(os.path.join(data_path, "*adj*.npy")))
    if candidates:
        return candidates[0]
    return ""


def get_config():
    parser = get_public_config()
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--node_num", type=int, default=0)
    parser.add_argument("--adj_path", type=str, default="")
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--experiment_timestamp", type=str, default="")

    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--llm_layer", type=int, default=6)
    parser.add_argument("--U", type=int, default=1)
    parser.add_argument("--steps_per_day", type=int, default=288)
    parser.add_argument("--auto_time_features", type=int, default=1)
    parser.add_argument("--add_time_of_day", type=int, default=1)
    parser.add_argument("--add_day_of_week", type=int, default=1)
    parser.add_argument("--time_start_offset", type=int, default=0)
    parser.add_argument("--time_day_idx", type=int, default=1)
    parser.add_argument("--day_in_week_idx", type=int, default=2)
    parser.add_argument("--gpt_model_name", type=str, default="gpt2")
    parser.add_argument("--gpt_local_files_only", type=int, default=1)
    parser.add_argument("--gpt_allow_download", type=int, default=1)
    parser.add_argument("--gpt_channel", type=int, default=256)
    parser.add_argument("--use_lora", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

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

    folder_name = build_experiment_dir_name(
        model_name=args.model_name,
        dataset=args.dataset,
        years=args.years,
        seq_len=args.seq_len,
        horizon=args.horizon,
        seed=args.seed,
        extra_parts=[
            ("ch", args.channels),
            ("llm", args.llm_layer),
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
        adj_path = str(args.adj_path).strip() or infer_adj_path(data_path)
        if not adj_path:
            raise ValueError(
                "adj_path is required for STLLM+ when using custom data_path, and no adjacency "
                "matrix could be inferred from {}".format(data_path)
            )
        logger.info("Use custom dataset path.")
        return data_path, node_num, adj_path

    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    return data_path, node_num, adj_path


def main():
    args, log_dir, logger, folder_name = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    data_path, node_num, adj_path = resolve_data_info(args, logger)
    adj_mx = load_adj_from_numpy(adj_path)

    if bool(args.auto_time_features):
        base_input_dim = args.input_dim
        add_time_of_day = bool(args.add_time_of_day)
        add_day_of_week = bool(args.add_day_of_week)
        extra_dim = int(add_time_of_day) + int(add_day_of_week)
        model_input_dim = base_input_dim + extra_dim

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

    model = STLLMPlus(
        node_num=node_num,
        input_dim=model_input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        adj_mx=adj_mx,
        channels=args.channels,
        llm_layer=args.llm_layer,
        U=args.U,
        steps_per_day=args.steps_per_day,
        time_day_idx=time_day_idx,
        day_in_week_idx=day_in_week_idx,
        gpt_model_name=args.gpt_model_name,
        gpt_local_files_only=args.gpt_local_files_only,
        gpt_allow_download=args.gpt_allow_download,
        gpt_channel=args.gpt_channel,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        dropout=args.dropout,
    )

    loss_fn = masked_mae
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = None

    engine = BaseEngine(
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
