# ST-LLM (ST_LLM) on standard LargeST subsets
# python experiments/stllm/main.py --device cuda:0 --dataset SD --years 2019 --model_name stllm --seed 2023 --bs 16 --seq_len 12 --horizon 12 --input_dim 3 --output_dim 1 --steps_per_day 288

# Sacramento flow-only example (no temporal channels: set idx to -1)
# python experiments/stllm/main.py --device cuda:0 --dataset Sacra --years 2023 --model_name stllm --seed 2023 --bs 16 --seq_len 12 --horizon 12 --input_dim 1 --output_dim 1 --data_path /path/to/data_root --node_num 517 --time_day_idx -1 --day_in_week_idx -1 --steps_per_day 288
