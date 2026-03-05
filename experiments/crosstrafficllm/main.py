import os
import numpy as np

import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

import torch

torch.set_num_threads(3)

from src.engines.crosstrafficllm_engine import CrossTrafficLLM_Engine
from src.models.crosstrafficllm import CrossTrafficLLM
from src.utils.args import get_public_config
from src.utils.dataloader import get_dataset_info
from src.utils.dataloader import load_adj_from_numpy
from src.utils.dataloader import load_dataset_with_reports
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

    parser.add_argument("--traffic_dim", type=int, default=3)
    parser.add_argument("--text_dim", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--text_hidden", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--graph_order", type=int, default=2)
    parser.add_argument("--patch_len", type=int, default=3)
    parser.add_argument("--top_k_text", type=int, default=4)
    parser.add_argument("--graph_fusion", type=float, default=0.7)

    parser.add_argument("--anomaly_topk", type=int, default=8)
    parser.add_argument("--report_len", type=int, default=16)
    parser.add_argument("--report_vocab_size", type=int, default=0)
    parser.add_argument("--memory_size", type=int, default=32)
    parser.add_argument("--report_prefix", type=str, default="report")
    parser.add_argument("--report_pad_id", type=int, default=0)

    parser.add_argument("--alignment_loss_weight", type=float, default=0.05)
    parser.add_argument("--report_loss_weight", type=float, default=0.2)

    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip_grad_value", type=float, default=5)
    args = parser.parse_args()

    total_dim = args.traffic_dim + args.text_dim
    if total_dim != args.input_dim:
        raise ValueError(
            "input_dim must equal traffic_dim + text_dim, got {} vs {}".format(
                args.input_dim, total_dim
            )
        )

    folder_name = "{}-{}-{}".format(args.dataset, args.traffic_dim, args.text_dim)
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

    dataloader, scaler = load_dataset_with_reports(
        data_path,
        args,
        logger,
        report_prefix=args.report_prefix,
    )

    model = CrossTrafficLLM(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        supports=supports,
        traffic_dim=args.traffic_dim,
        text_dim=args.text_dim,
        hidden_dim=args.hidden_dim,
        text_hidden=args.text_hidden,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        graph_order=args.graph_order,
        patch_len=args.patch_len,
        top_k_text=args.top_k_text,
        graph_fusion=args.graph_fusion,
        anomaly_topk=args.anomaly_topk,
        report_len=args.report_len,
        report_vocab_size=args.report_vocab_size,
        memory_size=args.memory_size,
        dropout=args.dropout,
    )

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = None

    engine = CrossTrafficLLM_Engine(
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
        alignment_loss_weight=args.alignment_loss_weight,
        report_loss_weight=args.report_loss_weight,
        report_pad_id=args.report_pad_id,
    )

    if args.mode == "train":
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()
