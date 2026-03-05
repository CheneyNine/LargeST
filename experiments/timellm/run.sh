# Time-LLM adaptation on standard LargeST subsets
# python experiments/timellm/main.py --device cuda:0 --dataset SD --years 2019 --model_name timellm --seed 2023 --bs 16 --seq_len 12 --horizon 12 --input_dim 3 --traffic_dim 3

# Flow-only setup (for custom subsets such as Sacramento)
# python experiments/timellm/main.py --device cuda:0 --dataset Sacra --years 2023 --model_name timellm --seed 2023 --bs 16 --seq_len 12 --horizon 12 --input_dim 1 --traffic_dim 1 --data_path /path/to/data_root --node_num 517
