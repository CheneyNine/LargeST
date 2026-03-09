import os
import numpy as np

import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

import torch

torch.set_num_threads(3)

from src.base.engine import BaseEngine
from src.models.timellm import TimeLLM
from src.utils.args import get_public_config
from src.utils.dataloader import get_dataset_info
from src.utils.dataloader import load_dataset
from src.utils.experiment_naming import build_experiment_dir_name
from src.utils.logging import get_logger
from src.utils.metrics import masked_mae


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--node_num", type=int, default=0)
    parser.add_argument("--run_tag", type=str, default="")

    parser.add_argument("--traffic_dim", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--llm_dim", type=int, default=768)
    parser.add_argument("--llm_layers", type=int, default=4)
    parser.add_argument("--llm_model", type=str, default="GPT2")  # GPT2, BERT, LLAMA
    parser.add_argument("--llm_model_name", type=str, default="")
    parser.add_argument("--llm_local_files_only", type=int, default=1)
    parser.add_argument("--llm_allow_download", type=int, default=1)
    parser.add_argument("--llm_torch_dtype", type=str, default="float32")
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--patch_len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--prompt_len", type=int, default=8)
    parser.add_argument("--prompt_mode", type=str, default="stats")  # stats, text
    parser.add_argument("--prompt_granularity", type=str, default="batch")  # batch, node
    parser.add_argument("--prompt_domain", type=int, default=0)
    parser.add_argument("--prompt_text", type=str, default="")
    parser.add_argument("--prompt_max_tokens", type=int, default=96)
    parser.add_argument("--prompt_batch_size", type=int, default=64)
    parser.add_argument("--node_chunk_size", type=int, default=32)
    parser.add_argument("--num_prototypes", type=int, default=1024)
    parser.add_argument("--top_k_lags", type=int, default=5)
    parser.add_argument("--freeze_backbone", type=int, default=1)
    parser.add_argument("--use_revin", type=int, default=1)

    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip_grad_value", type=float, default=5)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--train_sample_stride", type=int, default=1)
    parser.add_argument("--val_sample_stride", type=int, default=1)
    parser.add_argument("--test_sample_stride", type=int, default=1)
    parser.add_argument("--train_sample_limit", type=int, default=0)
    parser.add_argument("--val_sample_limit", type=int, default=0)
    parser.add_argument("--test_sample_limit", type=int, default=0)
    parser.add_argument("--use_swanlab", type=int, default=0)
    parser.add_argument("--swanlab_project", type=str, default="LargeST")
    parser.add_argument("--swanlab_experiment", type=str, default="")
    parser.add_argument("--swanlab_mode", type=str, default="cloud")
    args = parser.parse_args()

    if args.traffic_dim > args.input_dim:
        raise ValueError(
            "input_dim must be >= traffic_dim, got {} < {}".format(
                args.input_dim, args.traffic_dim
            )
        )

    folder_name = build_experiment_dir_name(
        dataset=args.dataset,
        years=args.years,
        seq_len=args.seq_len,
        horizon=args.horizon,
        seed=args.seed,
        extra_parts=[
            ("flow", args.traffic_dim),
            ("dm", args.d_model),
            ("llm", "{}{}".format(str(args.llm_model).lower(), int(args.llm_layers))),
            ("prompt", str(args.prompt_mode).lower()),
        ],
        run_tag=args.run_tag,
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

    model = TimeLLM(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        traffic_dim=args.traffic_dim,
        d_model=args.d_model,
        llm_dim=args.llm_dim,
        llm_layers=args.llm_layers,
        llm_model=args.llm_model,
        llm_model_name=args.llm_model_name,
        llm_local_files_only=args.llm_local_files_only,
        llm_allow_download=args.llm_allow_download,
        llm_torch_dtype=args.llm_torch_dtype,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        patch_len=args.patch_len,
        stride=args.stride,
        prompt_len=args.prompt_len,
        prompt_mode=args.prompt_mode,
        prompt_granularity=args.prompt_granularity,
        prompt_domain=args.prompt_domain,
        prompt_text=args.prompt_text,
        prompt_max_tokens=args.prompt_max_tokens,
        prompt_batch_size=args.prompt_batch_size,
        node_chunk_size=args.node_chunk_size,
        num_prototypes=args.num_prototypes,
        top_k_lags=args.top_k_lags,
        freeze_backbone=bool(args.freeze_backbone),
        use_revin=bool(args.use_revin),
        dropout=args.dropout,
        dataset_name=args.dataset,
    )

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

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
            else "{}-s{}".format(folder_name, args.seed),
            "mode": args.swanlab_mode,
            "logdir": log_dir,
            "config": vars(args),
        },
        log_interval=args.log_interval,
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
