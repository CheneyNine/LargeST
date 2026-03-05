#!/usr/bin/env python3
import argparse
import os

import numpy as np
import torch

from src.models.timecma import TimeCMA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--years", type=str, default="2019")
    parser.add_argument("--ts_dim", type=int, default=3)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--embedding_prefix", type=str, default="prompt_emb")
    parser.add_argument("--embedding_method", type=str, default="gpt2", choices=["gpt2", "stats"])
    parser.add_argument("--external_prompt_dim", type=int, default=768)
    parser.add_argument("--prompt_gen_model_name", type=str, default="gpt2")
    parser.add_argument("--prompt_gen_local_files_only", type=int, default=1)
    parser.add_argument("--prompt_gen_allow_download", type=int, default=1)
    parser.add_argument("--prompt_max_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def build_timecma_for_generator(args, node_num):
    model = TimeCMA(
        node_num=node_num,
        input_dim=args.ts_dim,
        output_dim=1,
        seq_len=args.seq_len,
        horizon=1,
        ts_dim=args.ts_dim,
        prompt_dim=0,
        channel=64,
        prompt_hidden=128,
        e_layer=1,
        d_layer=1,
        d_ff=256,
        head=8,
        dropout=0.1,
        prompt_pool="mean",
        external_prompt_dim=args.external_prompt_dim,
        prompt_gen_model_name=args.prompt_gen_model_name,
        prompt_gen_local_files_only=args.prompt_gen_local_files_only,
        prompt_gen_allow_download=args.prompt_gen_allow_download,
        prompt_max_tokens=args.prompt_max_tokens,
    )
    return model


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    year_dir = os.path.join(args.data_path, args.years)
    ptr = np.load(os.path.join(year_dir, "his.npz"))
    data = ptr["data"][..., : args.ts_dim]
    node_num = int(data.shape[1])

    model = build_timecma_for_generator(args, node_num=node_num).to(device)
    model.eval()

    x_offsets = np.arange(-(args.seq_len - 1), 1, 1)

    for split in ["train", "val", "test"]:
        idx = np.load(os.path.join(year_dir, "idx_{}.npy".format(split)))
        output_path = os.path.join(year_dir, "{}_{}.npy".format(args.embedding_prefix, split))
        output = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(idx), args.external_prompt_dim, node_num, 1),
        )

        for start in range(0, len(idx), args.batch_size):
            end = min(len(idx), start + args.batch_size)
            batch_idx = idx[start:end]
            history_index = batch_idx[:, None] + x_offsets[None, :]
            x_batch = data[history_index, :, :]  # [B, T, N, ts_dim]
            x_tensor = torch.tensor(x_batch, dtype=torch.float32, device=device)

            with torch.no_grad():
                emb = model.generate_prompt_embeddings(
                    x_tensor,
                    method=args.embedding_method,
                )
            output[start:end] = emb.detach().cpu().numpy().astype(np.float32)

            if start == 0:
                print(
                    "[{}] first batch embedding shape: {}".format(
                        split, tuple(emb.shape)
                    )
                )

        output.flush()
        print("Saved {} embeddings to {}".format(split, output_path))


if __name__ == "__main__":
    main()
