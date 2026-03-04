# TimeCMA ablation without prompt embeddings
# python experiments/timecma/main.py --device cuda:0 --dataset SD --years 2019 --model_name timecma --seed 2023 --bs 32 --input_dim 3 --ts_dim 3 --prompt_dim 0

# TimeCMA with prompt embeddings appended to his.npz
# Example: 3 traffic channels + 64 prompt channels
# python experiments/timecma/main.py --device cuda:0 --dataset SD --years 2019 --model_name timecma --seed 2023 --bs 16 --input_dim 67 --ts_dim 3 --prompt_dim 64 --prompt_hidden 128 --prompt_pool mean

# TimeCMA uses node-wise attention and is best suited to smaller LargeST subsets first.
