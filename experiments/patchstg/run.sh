# PatchSTG examples

# Sacramento 2023 flow-only 12->12
# DESC="Sacra2023 PatchSTG 12到12 基线" python experiments/patchstg/main.py \
#   --device cuda:0 \
#   --dataset Sacra \
#   --years 2023 \
#   --model_name patchstg \
#   --data_path /root/XTraffic/data/processed_y2023_sacramento \
#   --meta_path /root/XTraffic/data/sensor_meta_feature.csv \
#   --node_order_path /root/XTraffic/data/processed_y2023_sacramento/node_order_sacramento.npy \
#   --node_num 517 \
#   --seq_len 12 \
#   --horizon 12 \
#   --traffic_dim 1 \
#   --steps_per_day 288 \
#   --bs 16 \
#   --seed 2023 \
#   --desc "$DESC"

# Sacramento January flow-only 24->24
# DESC="SacraJan2023 PatchSTG 24到24 基线" python experiments/patchstg/main.py \
#   --device cuda:0 \
#   --dataset SacraJan \
#   --years 2023 \
#   --model_name patchstg \
#   --data_path /root/XTraffic/data/processed_y2023_sacramento_jan_q24h24 \
#   --meta_path /root/XTraffic/data/sensor_meta_feature.csv \
#   --node_order_path /root/XTraffic/data/processed_y2023_sacramento_jan_q24h24/node_order_sacramento.npy \
#   --node_num 517 \
#   --seq_len 24 \
#   --horizon 24 \
#   --traffic_dim 1 \
#   --steps_per_day 288 \
#   --bs 16 \
#   --seed 2023 \
#   --desc "$DESC"
