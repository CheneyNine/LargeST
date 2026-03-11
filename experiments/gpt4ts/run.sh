# GPT4TS examples

# Sacramento flow-only example
# DESC="Sacra2023 GPT4TS 12预测12 flow" python experiments/gpt4ts/main.py \
#   --device cuda:0 \
#   --dataset Sacra \
#   --years 2023 \
#   --model_name gpt4ts \
#   --seed 2023 \
#   --bs 16 \
#   --seq_len 12 \
#   --horizon 12 \
#   --input_dim 1 \
#   --traffic_dim 1 \
#   --data_path /root/XTraffic/data/processed_y2023_sacramento \
#   --node_num 517 \
#   --desc "$DESC"
