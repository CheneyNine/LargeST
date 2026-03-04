import os
import numpy as np

import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

import torch

torch.set_num_threads(3)

from src.engines.e2cstp_engine import E2CSTP_Engine
from src.models.e2cstp import E2CSTP
from src.utils.args import get_public_config
from src.utils.dataloader import get_dataset_info
from src.utils.dataloader import load_adj_from_numpy
from src.utils.dataloader import load_dataset
from src.utils.graph_algo import normalize_adj_mx
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
    parser.add_argument("--adj_type", type=str, default="doubletransition")

    parser.add_argument("--st_dim", type=int, default=3)
    parser.add_argument("--text_dim", type=int, default=0)
    parser.add_argument("--image_dim", type=int, default=0)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--encoder_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mamba_expand", type=int, default=2)
    parser.add_argument("--mamba_kernel", type=int, default=3)
    parser.add_argument("--gcn_order", type=int, default=2)
    parser.add_argument("--decoder_hidden", type=int, default=256)

    parser.add_argument("--graph_fusion", type=float, default=0.7)
    parser.add_argument("--causal_momentum", type=float, default=0.9)
    parser.add_argument("--causal_update_interval", type=int, default=50)
    parser.add_argument("--intervention_scale", type=float, default=0.5)
    parser.add_argument("--branch_alpha", type=float, default=0.5)
    parser.add_argument("--causal_reg_weight", type=float, default=0.05)

    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip_grad_value", type=float, default=5)
    args = parser.parse_args()

    total_modal_dim = args.st_dim + args.text_dim + args.image_dim
    if total_modal_dim != args.input_dim:
        raise ValueError(
            "input_dim must equal st_dim + text_dim + image_dim, got {} vs {}".format(
                args.input_dim, total_modal_dim
            )
        )

    folder_name = "{}-{}-{}-{}".format(
        args.dataset, args.st_dim, args.text_dim, args.image_dim
    )
    log_dir = "./experiments/{}/{}/".format(args.model_name, folder_name)
    logger = get_logger(log_dir, __name__, "record_s{}.log".format(args.seed))
    logger.info(args)

    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info("Adj path: " + adj_path)

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, args.adj_type)
    supports = [torch.tensor(adj, dtype=torch.float32, device=device) for adj in adj_mx]

    dataloader, scaler = load_dataset(data_path, args, logger)

    model = E2CSTP(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        supports=supports,
        st_dim=args.st_dim,
        text_dim=args.text_dim,
        image_dim=args.image_dim,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        num_heads=args.num_heads,
        mamba_expand=args.mamba_expand,
        mamba_kernel=args.mamba_kernel,
        gcn_order=args.gcn_order,
        decoder_hidden=args.decoder_hidden,
        graph_fusion=args.graph_fusion,
        causal_momentum=args.causal_momentum,
        causal_update_interval=args.causal_update_interval,
        intervention_scale=args.intervention_scale,
        dropout=args.dropout,
    )

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = None

    engine = E2CSTP_Engine(
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
        branch_alpha=args.branch_alpha,
        causal_reg_weight=args.causal_reg_weight,
    )

    if args.mode == "train":
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()
