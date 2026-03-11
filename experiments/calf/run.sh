# CALF examples

# Sacramento flow-only example
# DESC="Sacra2023 CALF 12预测12 flow" python experiments/calf/main.py \
#   --device cuda:0 \
#   --dataset Sacra \
#   --years 2023 \
#   --model_name calf \
#   --seed 2023 \
#   --bs 8 \
#   --seq_len 12 \
#   --horizon 12 \
#   --input_dim 1 \
#   --traffic_dim 1 \
#   --data_path /root/XTraffic/data/processed_y2023_sacramento \
#   --node_num 517 \
#   --word_embedding_path ./cache/calf_wte_pca_500.pt \
#   --desc "$DESC"
