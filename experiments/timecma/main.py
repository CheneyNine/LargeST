import os
import numpy as np

import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

import torch

torch.set_num_threads(3)

from src.engines.timecma_engine import TimeCMA_Engine
from src.models.timecma import TimeCMA
from src.utils.args import get_public_config
from src.utils.dataloader import get_dataset_info
from src.utils.dataloader import load_dataset
from src.utils.dataloader import load_dataset_with_embeddings
from src.utils.experiment_naming import build_experiment_dir_name
from src.utils.logging import get_logger
from src.utils.metrics import masked_mse


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.set_defaults(bs=16, max_epochs=999, patience=50, seed=2024)
    parser.add_argument("--ts_dim", type=int, default=3)
    parser.add_argument("--prompt_dim", type=int, default=0)
    parser.add_argument("--channel", type=int, default=64)
    parser.add_argument("--prompt_hidden", type=int, default=128)
    parser.add_argument("--e_layer", type=int, default=2)
    parser.add_argument("--d_layer", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=32)
    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--prompt_pool", type=str, default="mean", choices=["mean", "last", "max"])
    parser.add_argument("--d_llm", type=int, default=768)
    parser.add_argument("--external_prompt_dim", type=int, default=768)
    parser.add_argument("--prompt_gen_model_name", type=str, default="gpt2")
    parser.add_argument("--prompt_gen_local_files_only", type=int, default=1)
    parser.add_argument("--prompt_gen_allow_download", type=int, default=1)
    parser.add_argument("--prompt_max_tokens", type=int, default=512)
    parser.add_argument("--prompt_data_name", type=str, default="")
    parser.add_argument("--prompt_input_template", type=str, default="")
    parser.add_argument("--prompt_freq_minutes", type=int, default=5)
    parser.add_argument("--prompt_start_datetime", type=str, default="")
    parser.add_argument("--use_external_embeddings", type=int, default=0)
    parser.add_argument("--embedding_prefix", type=str, default="prompt_emb")
    parser.add_argument("--generate_embeddings_on_the_fly", type=int, default=0)
    parser.add_argument("--embedding_method", type=str, default="gpt2", choices=["gpt2", "stats"])
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--node_num", type=int, default=-1)
    parser.add_argument("--run_tag", type=str, default="")

    parser.add_argument("--lrate", type=float, default=1e-4)
    parser.add_argument("--wdecay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--clip_grad_value", type=float, default=5)
    args = parser.parse_args()

    total_dim = args.ts_dim + args.prompt_dim
    if total_dim != args.input_dim:
        raise ValueError(
            "input_dim must equal ts_dim + prompt_dim, got {} vs {}".format(
                args.input_dim, total_dim
            )
        )
    if int(args.external_prompt_dim) != int(args.d_llm):
        raise ValueError(
            "external_prompt_dim must equal d_llm for official-style TimeCMA, got {} vs {}".format(
                args.external_prompt_dim, args.d_llm
            )
        )
    if not str(args.prompt_data_name).strip():
        args.prompt_data_name = args.dataset
    if not str(args.prompt_start_datetime).strip():
        year_token = str(args.years).split("_")[0]
        if year_token.isdigit() and len(year_token) == 4:
            args.prompt_start_datetime = "{}-01-01 00:00:00".format(year_token)
        else:
            args.prompt_start_datetime = "2019-01-01 00:00:00"

    folder_name = build_experiment_dir_name(
        dataset=args.dataset,
        years=args.years,
        seq_len=args.seq_len,
        horizon=args.horizon,
        seed=args.seed,
        extra_parts=[
            ("ts", args.ts_dim),
            ("pd", args.prompt_dim),
            ("ch", args.channel),
            ("el", args.e_layer),
            ("dl", args.d_layer),
            ("lr", args.lrate),
            ("drop", args.dropout),
        ],
        run_tag=args.run_tag,
    )
    log_dir = "./experiments/{}/{}/".format(args.model_name, folder_name)
    logger = get_logger(log_dir, __name__, "record_s{}.log".format(args.seed))
    logger.info(args)
    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    if str(args.data_path).strip():
        data_path = args.data_path
        if args.node_num > 0:
            node_num = int(args.node_num)
        else:
            ptr = np.load(os.path.join(data_path, args.years, "his.npz"))
            node_num = int(ptr["data"].shape[1])
    else:
        data_path, _, node_num = get_dataset_info(args.dataset)
    if bool(args.use_external_embeddings) or bool(args.generate_embeddings_on_the_fly):
        dataloader, scaler = load_dataset_with_embeddings(
            data_path, args, logger, embedding_prefix=args.embedding_prefix
        )
    else:
        dataloader, scaler = load_dataset(data_path, args, logger)

    model = TimeCMA(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        ts_dim=args.ts_dim,
        prompt_dim=args.prompt_dim,
        channel=args.channel,
        prompt_hidden=args.prompt_hidden,
        e_layer=args.e_layer,
        d_layer=args.d_layer,
        d_ff=args.d_ff,
        head=args.head,
        dropout=args.dropout,
        prompt_pool=args.prompt_pool,
        external_prompt_dim=args.external_prompt_dim,
        prompt_gen_model_name=args.prompt_gen_model_name,
        prompt_gen_local_files_only=args.prompt_gen_local_files_only,
        prompt_gen_allow_download=args.prompt_gen_allow_download,
        prompt_max_tokens=args.prompt_max_tokens,
        prompt_data_name=args.prompt_data_name,
        prompt_input_template=args.prompt_input_template,
        prompt_freq_minutes=args.prompt_freq_minutes,
    )

    loss_fn = masked_mse
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=min(int(args.max_epochs), 50), eta_min=1e-6
    )

    engine = TimeCMA_Engine(
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
        generate_embeddings_on_the_fly=args.generate_embeddings_on_the_fly,
        embedding_method=args.embedding_method,
        prompt_start_datetime=args.prompt_start_datetime,
        prompt_freq_minutes=args.prompt_freq_minutes,
    )

    if args.mode == "train":
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()
