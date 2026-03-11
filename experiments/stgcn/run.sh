# STGCN examples

# Standard subsets
# python experiments/stgcn/main.py --device cuda:0 --dataset CA --years 2019 --model_name stgcn --seed 2023 --bs 64
# python experiments/stgcn/main.py --device cuda:0 --dataset GLA --years 2019 --model_name stgcn --seed 2023 --bs 64
# python experiments/stgcn/main.py --device cuda:0 --dataset GBA --years 2019 --model_name stgcn --seed 2023 --bs 64
# python experiments/stgcn/main.py --device cuda:0 --dataset SD --years 2019 --model_name stgcn --seed 2023 --bs 64

# Sacramento example
# DESC="Sacra2023 STGCN 12预测12 flow" python experiments/stgcn/main.py \
#   --device cuda:0 \
#   --dataset Sacra \
#   --years 2023 \
#   --model_name stgcn \
#   --seed 2023 \
#   --bs 32 \
#   --seq_len 12 \
#   --horizon 12 \
#   --input_dim 1 \
#   --data_path /root/XTraffic/data/processed_y2023_sacramento \
#   --node_num 517 \
#   --adj_path /root/XTraffic/data/processed_y2023_sacramento/adj_matrix_sacramento.npy \
#   --desc "$DESC"
