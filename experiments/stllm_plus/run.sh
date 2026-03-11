# STLLM+ examples

# Sacramento flow-only example with auto-generated time/day features
# DESC="Sacra2023 STLLM+ 12预测12 flow" python experiments/stllm_plus/main.py \
#   --device cuda:0 \
#   --dataset Sacra \
#   --years 2023 \
#   --model_name stllm_plus \
#   --seed 2023 \
#   --bs 8 \
#   --seq_len 12 \
#   --horizon 12 \
#   --input_dim 1 \
#   --data_path /root/XTraffic/data/processed_y2023_sacramento \
#   --node_num 517 \
#   --adj_path /root/XTraffic/data/processed_y2023_sacramento/adj_matrix_sacramento.npy \
#   --auto_time_features 1 \
#   --desc "$DESC"
