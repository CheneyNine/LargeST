#!/bin/bash

DESC="${DESC:-STEVE baseline}"

# STEVE on LargeST single-modal traffic features
# python experiments/steve/main.py --device cuda:0 --dataset SD --years 2019 --model_name steve --seed 2023 --bs 16 --input_dim 3 --traffic_dim 3 --embed_dim 64 --desc "$DESC"

# Larger subset example
# python experiments/steve/main.py --device cuda:0 --dataset GLA --years 2019 --model_name steve --seed 2023 --bs 8 --input_dim 3 --traffic_dim 3 --embed_dim 64 --spatial_sample_size 256 --mi_sample_size 2048 --desc "$DESC"
