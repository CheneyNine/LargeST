# TimeCMA paper-style pipeline in LargeST
# Official-aligned defaults:
#   seed=2024, bs=16, channel=64, d_ff=32, lr=1e-4, weight_decay=1e-3,
#   dropout=0.5, AdamW, CosineAnnealingLR, patience=50
#
# The paper/official repo first stores GPT2 last-token embeddings offline, then
# trains TimeCMA with those cached embeddings. Prompt text is constructed from
# raw history values, not from normalized inputs.
#
# 1) Precompute prompt embeddings (official-style offline storage):
# python scripts/generate_timecma_prompt_embeddings.py \
#   --data_path /root/XTraffic/data/processed_y2023_sacramento_jan_q12h12 \
#   --years 2023 \
#   --ts_dim 1 \
#   --seq_len 12 \
#   --batch_size 1 \
#   --embedding_method gpt2 \
#   --d_llm 768 \
#   --external_prompt_dim 768 \
#   --data_name SACRA \
#   --start_datetime '2023-01-01 00:00:00' \
#   --freq_minutes 5 \
#   --device cuda:0
#
# 2) Train with cached prompt_emb_{train,val,test}.npy:
# python experiments/timecma/main.py \
#   --device cuda:0 \
#   --dataset SacraJan \
#   --years 2023 \
#   --model_name timecma \
#   --data_path /root/XTraffic/data/processed_y2023_sacramento_jan_q12h12 \
#   --node_num 517 \
#   --seq_len 12 \
#   --horizon 12 \
#   --input_dim 1 \
#   --ts_dim 1 \
#   --prompt_dim 0 \
#   --use_external_embeddings 1 \
#   --embedding_prefix prompt_emb \
#   --d_llm 768 \
#   --external_prompt_dim 768 \
#   --prompt_data_name SACRA \
#   --prompt_start_datetime '2023-01-01 00:00:00' \
#   --prompt_freq_minutes 5 \
#   --run_tag official_gpt2_external
