# CrossTrafficLLM numeric forecasting only
# python experiments/crosstrafficllm/main.py --device cuda:0 --dataset SD --years 2019 --model_name crosstrafficllm --seed 2023 --bs 16 --input_dim 3 --traffic_dim 3 --text_dim 0

# CrossTrafficLLM with aligned text embeddings appended to his.npz
# Example: 3 traffic channels + 32 text embedding channels
# python experiments/crosstrafficllm/main.py --device cuda:0 --dataset SD --years 2019 --model_name crosstrafficllm --seed 2023 --bs 8 --input_dim 35 --traffic_dim 3 --text_dim 32 --hidden_dim 64 --text_hidden 64

# Optional interpretable report generation:
# put sample-aligned report targets at data/<dataset>/<year>/report_{train,val,test}.npy
# then set --report_vocab_size and related report arguments
# python experiments/crosstrafficllm/main.py --device cuda:0 --dataset SD --years 2019 --model_name crosstrafficllm --seed 2023 --bs 8 --input_dim 35 --traffic_dim 3 --text_dim 32 --report_vocab_size 5000 --report_len 16
