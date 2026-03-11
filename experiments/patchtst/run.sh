# PatchTST examples

# Sacramento 2023 flow-only 12->12
# DESC="Sacra2023 PatchTST q12->12 flow baseline" python experiments/patchtst/main.py \
#   --device cuda:0 \
#   --dataset Sacra \
#   --years 2023 \
#   --model_name patchtst \
#   --data_path /root/XTraffic/data/processed_y2023_sacramento \
#   --node_num 517 \
#   --seq_len 12 \
#   --horizon 12 \
#   --input_dim 1 \
#   --traffic_dim 1 \
#   --d_model 128 \
#   --n_heads 8 \
#   --e_layers 3 \
#   --d_ff 256 \
#   --patch_len 4 \
#   --stride 2 \
#   --bs 16 \
#   --seed 2023 \
#   --desc "$DESC"

# Sacramento January flow-only 24->24
# DESC="SacraJan2023 PatchTST q24->24 flow baseline" python experiments/patchtst/main.py \
#   --device cuda:0 \
#   --dataset SacraJan \
#   --years 2023 \
#   --model_name patchtst \
#   --data_path /root/XTraffic/data/processed_y2023_sacramento_jan_q24h24 \
#   --node_num 517 \
#   --seq_len 24 \
#   --horizon 24 \
#   --input_dim 1 \
#   --traffic_dim 1 \
#   --d_model 128 \
#   --n_heads 8 \
#   --e_layers 3 \
#   --d_ff 256 \
#   --patch_len 8 \
#   --stride 4 \
#   --bs 16 \
#   --seed 2023 \
#   --desc "$DESC"
