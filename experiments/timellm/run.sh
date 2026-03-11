#!/bin/bash

DESC="${DESC:-TimeLLM baseline}"

# Time-LLM integration on standard LargeST subsets (efficient stats-prompt mode)
# python experiments/timellm/main.py --device cuda:0 --dataset SD --years 2019 --model_name timellm --seed 2023 --bs 16 --seq_len 12 --horizon 12 --input_dim 3 --traffic_dim 3 --llm_model GPT2 --llm_local_files_only 1 --llm_allow_download 1 --prompt_mode stats --prompt_granularity batch --run_tag sd_stats --desc "$DESC"

# Sacramento + LLAMA 32-layer, 12 -> 12 (flow only), with SwanLab logging
# export HF_ENDPOINT=https://hf-mirror.com
# python experiments/timellm/main.py --device cuda:0 --dataset Sacra --years 2023 --model_name timellm --seed 2023 --bs 1 --seq_len 12 --horizon 12 --input_dim 1 --traffic_dim 1 --data_path /root/XTraffic/data/processed_y2023_sacramento --node_num 517 --llm_model LLAMA --llm_layers 32 --llm_model_name huggyllama/llama-7b --llm_local_files_only 1 --llm_allow_download 1 --llm_torch_dtype float16 --prompt_mode stats --prompt_granularity batch --node_chunk_size 8 --train_sample_stride 48 --val_sample_stride 12 --test_sample_stride 12 --log_interval 10 --freeze_backbone 1 --run_tag sacra_llama32_12to12 --use_swanlab 1 --swanlab_project LargeST --swanlab_experiment timellm_sacra_llama32_12to12 --desc "$DESC"

# Sacramento + LLAMA 32-layer, 24 -> 24 (flow only), with SwanLab logging
# export HF_ENDPOINT=https://hf-mirror.com
# python experiments/timellm/main.py --device cuda:0 --dataset Sacra --years 2023 --model_name timellm --seed 2023 --bs 1 --seq_len 24 --horizon 24 --input_dim 1 --traffic_dim 1 --data_path /root/XTraffic/data/processed_y2023_sacramento --node_num 517 --llm_model LLAMA --llm_layers 32 --llm_model_name huggyllama/llama-7b --llm_local_files_only 1 --llm_allow_download 1 --llm_torch_dtype float16 --prompt_mode stats --prompt_granularity batch --node_chunk_size 8 --train_sample_stride 48 --val_sample_stride 12 --test_sample_stride 12 --log_interval 10 --freeze_backbone 1 --run_tag sacra_llama32_24to24 --use_swanlab 1 --swanlab_project LargeST --swanlab_experiment timellm_sacra_llama32_24to24 --desc "$DESC"

# Official-style text prompt mode (heavier, usually for smaller settings)
# python experiments/timellm/main.py --device cuda:0 --dataset SD --years 2019 --model_name timellm --seed 2023 --bs 4 --seq_len 12 --horizon 12 --input_dim 3 --traffic_dim 3 --llm_model GPT2 --prompt_mode text --prompt_granularity batch --prompt_max_tokens 64 --run_tag sd_text --desc "$DESC"
