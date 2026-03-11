#!/bin/bash

DESC="${DESC:-E2CSTP baseline}"

# LargeST single-modal usage (default: only spatio-temporal channels)
# python experiments/e2cstp/main.py --device cuda:0 --dataset GLA --years 2019 --model_name e2cstp --seed 2023 --bs 8 --input_dim 3 --st_dim 3 --text_dim 0 --image_dim 0 --desc "$DESC"
# python experiments/e2cstp/main.py --device cuda:0 --dataset GBA --years 2019 --model_name e2cstp --seed 2023 --bs 16 --input_dim 3 --st_dim 3 --text_dim 0 --image_dim 0 --desc "$DESC"
# python experiments/e2cstp/main.py --device cuda:0 --dataset SD --years 2019 --model_name e2cstp --seed 2023 --bs 64 --input_dim 3 --st_dim 3 --text_dim 0 --image_dim 0 --desc "$DESC"

# Example for aligned multi-modal features appended to his.npz:
# python experiments/e2cstp/main.py --device cuda:0 --dataset SD --years 2019 --model_name e2cstp --seed 2023 --bs 32 --input_dim 19 --st_dim 3 --text_dim 8 --image_dim 8 --desc "$DESC"
