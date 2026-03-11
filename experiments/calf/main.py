import os
import numpy as np

import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

import torch

torch.set_num_threads(3)

from src.engines.calf_engine import CALFEngine
from src.models.calf import CALF
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
    parser.set_defaults(bs=8, max_epochs=10, patience=5)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--node_num", type=int, default=0)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--experiment_timestamp", type=str, default="")

    parser.add_argument("--traffic_dim", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--gpt_layers", type=int, default=6)
    parser.add_argument("--pca_encoder_layers", type=int, default=1)
    parser.add_argument("--pretrain", type=int, default=1)
    parser.add_argument("--use_lora", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--gpt_model_name", type=str, default="gpt2")
    parser.add_argument("--gpt_local_files_only", type=int, default=1)
    parser.add_argument("--gpt_allow_download", type=int, default=1)
    parser.add_argument("--word_embedding_path", type=str, default="./cache/calf_wte_pca_500.pt")
    parser.add_argument("--word_embedding_components", type=int, default=500)

    parser.add_argument("--lrate", type=float, default=5e-4)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--clip_grad_value", type=float, default=5)
    parser.add_argument("--feature_loss", type=str, default="smooth_l1", choices=["l1", "smooth_l1", "mse"])
    parser.add_argument("--output_loss", type=str, default="smooth_l1", choices=["l1", "smooth_l1", "mse"])
    parser.add_argument("--task_loss", type=str, default="smooth_l1", choices=["l1", "smooth_l1", "mse"])
    parser.add_argument("--feature_w", type=float, default=0.01)
    parser.add_argument("--output_w", type=float, default=1.0)
    parser.add_argument("--task_w", type=float, default=1.0)

    parser.add_argument("--use_swanlab", type=int, default=1)
    parser.add_argument("--swanlab_project", type=str, default="LargeST")
    parser.add_argument("--swanlab_experiment", type=str, default="")
    parser.add_argument("--swanlab_mode", type=str, default="cloud")
    parser.add_argument("--swanlab_description", type=str, default="")
    parser.add_argument("--desc", dest="swanlab_description", type=str)
    parser.add_argument("--swanlab_lark_webhook_url", type=str, default=os.getenv("SWANLAB_LARK_WEBHOOK_URL", ""))
    parser.add_argument("--swanlab_lark_secret", type=str, default=os.getenv("SWANLAB_LARK_SECRET", ""))
    args = parser.parse_args()

    if args.traffic_dim != 1:
        raise ValueError("CALF is integrated as flow-only forecasting, set traffic_dim=1.")

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

    model = CALF(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        traffic_dim=args.traffic_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        gpt_layers=args.gpt_layers,
        pca_encoder_layers=args.pca_encoder_layers,
        pretrain=args.pretrain,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gpt_model_name=args.gpt_model_name,
        gpt_local_files_only=args.gpt_local_files_only,
        gpt_allow_download=args.gpt_allow_download,
        word_embedding_path=args.word_embedding_path,
        word_embedding_components=args.word_embedding_components,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=min(int(args.max_epochs), 10), eta_min=1e-8
    )

    engine = CALFEngine(
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
        feature_loss=args.feature_loss,
        output_loss=args.output_loss,
        task_loss=args.task_loss,
        feature_w=args.feature_w,
        output_w=args.output_w,
        task_w=args.task_w,
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
