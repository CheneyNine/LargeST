# TimeCMA ablation without prompt embeddings
# python experiments/timecma/main.py --device cuda:0 --dataset SD --years 2019 --model_name timecma --seed 2023 --bs 32 --input_dim 3 --ts_dim 3 --prompt_dim 0

# TimeCMA with prompt embeddings appended to his.npz
# Example: 3 traffic channels + 64 prompt channels
# python experiments/timecma/main.py --device cuda:0 --dataset SD --years 2019 --model_name timecma --seed 2023 --bs 16 --input_dim 67 --ts_dim 3 --prompt_dim 64 --prompt_hidden 128 --prompt_pool mean

# TimeCMA with external prompt embeddings (official-style):
# 1) Precompute embeddings:
# python scripts/generate_timecma_prompt_embeddings.py --data_path data/sd --years 2019 --ts_dim 3 --seq_len 12 --embedding_method gpt2 --device cuda:0
# 2) Train with external embeddings prompt_emb_{train,val,test}.npy
# python experiments/timecma/main.py --device cuda:0 --dataset SD --years 2019 --model_name timecma --seed 2023 --bs 16 --input_dim 3 --ts_dim 3 --prompt_dim 0 --use_external_embeddings 1 --embedding_prefix prompt_emb --external_prompt_dim 768

# TimeCMA with online embedding generation (no pre-save files)
# python experiments/timecma/main.py --device cuda:0 --dataset SD --years 2019 --model_name timecma --seed 2023 --bs 8 --input_dim 3 --ts_dim 3 --prompt_dim 0 --generate_embeddings_on_the_fly 1 --embedding_method gpt2 --external_prompt_dim 768

# TimeCMA uses node-wise attention and is best suited to smaller LargeST subsets first.
