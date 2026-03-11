# LargeST Research Workspace

This repository is a research-oriented fork of the original [LargeST](https://arxiv.org/abs/2306.08259) codebase. It is no longer maintained as a pure benchmark mirror. The current project is focused on traffic forecasting experiments over custom XTraffic data, especially Sacramento full-year, January-only, and incident-aware subsets, while keeping the original LargeST training pipeline as the base framework.

## What This Repository Is For

- Run unified traffic forecasting experiments with a shared dataloader, engine, logging, and evaluation pipeline.
- Support both original graph baselines and adapted recent models in the same framework.
- Work with Sacramento/XTraffic datasets in `his.npz` format and incident-aware auxiliary tensors.
- Reproduce controlled experiments such as:
  - full-year vs January
  - event vs no-event
  - flow-only vs extended feature settings
  - official-style prompt embedding pipelines for TimeCMA
- Track experiments with SwanLab and keep run outputs in consistent experiment folders.

## Current Scope

The active focus of this project is not the original CA / GLA / GBA / SD benchmark release. The active workflow is centered on custom datasets derived from XTraffic, especially:

| Dataset Name | Description | Typical Shape |
| --- | --- | --- |
| `Sacra2023` | Sacramento full-year flow dataset | `his.npz: (105120, 517, 1)` |
| `SacraJan2023` | January-only Sacramento subset | `his.npz: (8928, 517, 1)` |
| `SacraEvent2023` | Forecast windows whose prediction horizon contains at least one incident | same base tensor, filtered indices |
| `SacraNoEvent2023` | Forecast windows whose prediction horizon contains no incident | same base tensor, filtered indices |

The repository assumes the standard training tensor format:

```text
his.npz["data"] -> [T, N, F]
```

For incident-aware analysis, an auxiliary tensor may also exist:

```text
traffic_incident_ntf.npy -> [N, T, F]
```

where `F` commonly includes:

- `flow`
- `occupancy`
- `speed`
- `incident_id`
- `incident_progress`

## Supported Models

This repository currently contains both original LargeST baselines and project-specific integrations.

### Original / classical baselines

- Historical Last
- LSTM
- DCRNN
- AGCRN
- STGCN
- GWNET
- ASTGCN
- STTN
- STGODE
- DSTAGNN
- DGCRN
- D2STGNN

### Adapted or integrated models

- STEVE
- E2CSTP
- TimeCMA
- TimeLLM
- ST-LLM
- CrossTrafficLLM
- PatchTST

Not every integrated model should be interpreted as an official reproduction of the original authors' released code. Some are framework-compatible adaptations designed to fit the native LargeST data pipeline and experiment engine.

## Repository Layout

```text
experiments/          model-specific entrypoints
src/models/           model definitions
src/engines/          custom training/evaluation engines
src/utils/            dataloader, naming, SwanLab, metrics, graph utils
scripts/              preprocessing and experiment helper scripts
data/                 original LargeST data helpers
docs/                 project notes for experiment organization
```

## Experiment Conventions

### Run naming

All new formal runs should use:

```text
<model>_<dataset>_q<seq_len>_h<horizon>_s<seed>_t<YYMMDDHHMM>
```

Example:

```text
timecma_SacraJan2023_q24_h24_s2024_t2603091230
```

### Run directory layout

Each run should be stored under:

```text
experiments/<model>/<run_dir>/
```

with a flattened structure such as:

```text
record_s<seed>.log
final_model_s<seed>.pt
pipeline.log
pipeline.pid
run_pipeline.sh
```

### SwanLab

This project uses SwanLab as the default experiment tracker for new formal runs.

Recommended flags:

```bash
--use_swanlab 1 \
--swanlab_project LargeST \
--experiment_timestamp 2603101027 \
--desc "SacraJan2023 PatchTST 12->12 baseline"
```

`--desc` is the short alias of `--swanlab_description`.

## Environment

Recommended environment for current experiments:

- Python 3.10
- PyTorch with CUDA
- `TSLib310` on the training server

The project has been validated with SwanLab and Lark notifications in the Python 3.10 environment. Older Python 3.8 environments may fail on some SwanLab plugin imports.

## Quick Start

### 1. STEVE on Sacramento full-year flow-only data

```bash
python experiments/steve/main.py \
  --device cuda:0 \
  --dataset Sacra \
  --years 2023 \
  --model_name steve \
  --data_path /root/XTraffic/data/processed_y2023_sacramento \
  --adj_path /root/XTraffic/data/processed_y2023_sacramento/adj_matrix_sacramento.npy \
  --node_num 517 \
  --seq_len 12 \
  --horizon 12 \
  --input_dim 1 \
  --traffic_dim 1 \
  --seed 2023 \
  --use_swanlab 1 \
  --swanlab_project LargeST \
  --desc "Sacra2023 STEVE 12->12 baseline"
```

### 2. E2CSTP on Sacramento full-year flow-only data

```bash
python experiments/e2cstp/main.py \
  --device cuda:0 \
  --dataset Sacra \
  --years 2023 \
  --model_name e2cstp \
  --data_path /root/XTraffic/data/processed_y2023_sacramento \
  --adj_path /root/XTraffic/data/processed_y2023_sacramento/adj_matrix_sacramento.npy \
  --node_num 517 \
  --seq_len 12 \
  --horizon 12 \
  --input_dim 1 \
  --st_dim 1 \
  --text_dim 0 \
  --image_dim 0 \
  --seed 2023 \
  --use_swanlab 1 \
  --swanlab_project LargeST \
  --desc "Sacra2023 E2CSTP 12->12 baseline"
```

### 3. PatchTST on Sacramento full-year flow-only data

```bash
python experiments/patchtst/main.py \
  --device cuda:0 \
  --dataset Sacra \
  --years 2023 \
  --model_name patchtst \
  --data_path /root/XTraffic/data/processed_y2023_sacramento \
  --node_num 517 \
  --seq_len 12 \
  --horizon 12 \
  --input_dim 1 \
  --traffic_dim 1 \
  --seed 2023 \
  --use_swanlab 1 \
  --swanlab_project LargeST \
  --desc "Sacra2023 PatchTST 12->12 baseline"
```

### 4. TimeCMA official-style pipeline

TimeCMA should be run in two stages when reproducing the official prompt-embedding workflow.

Stage 1: generate GPT2 prompt embeddings

```bash
python scripts/generate_timecma_prompt_embeddings.py \
  --data_path /root/XTraffic/data/processed_y2023_sacramento_jan_q12h12 \
  --years 2023 \
  --seq_len 12 \
  --embedding_method gpt2 \
  --d_llm 768 \
  --external_prompt_dim 768 \
  --data_name SACRA \
  --start_datetime "2023-01-01 00:00:00" \
  --freq_minutes 5 \
  --device cuda:0
```

Stage 2: train with external embeddings

```bash
python experiments/timecma/main.py \
  --device cuda:0 \
  --dataset SacraJan \
  --years 2023 \
  --model_name timecma \
  --data_path /root/XTraffic/data/processed_y2023_sacramento_jan_q12h12 \
  --node_num 517 \
  --seq_len 12 \
  --horizon 12 \
  --input_dim 1 \
  --ts_dim 1 \
  --prompt_dim 0 \
  --use_external_embeddings 1 \
  --embedding_prefix prompt_emb \
  --d_llm 768 \
  --external_prompt_dim 768 \
  --use_swanlab 1 \
  --swanlab_project LargeST \
  --desc "SacraJan2023 TimeCMA GPT2 external 12->12"
```

### 5. TimeLLM

TimeLLM is available in the framework, but large LLM backbones can be numerically and computationally fragile on large node sets. For practical runs, use conservative settings and monitor stability carefully.

## Incident-Aware Data Splitting

To split Sacramento full-year data into event and no-event windows based on `incident_id` in the prediction horizon:

```bash
python scripts/split_sacra_by_incident_windows.py \
  --input_root /root/XTraffic/data/processed_y2023_sacramento \
  --year 2023 \
  --seq_len 12 \
  --horizon 12 \
  --criterion forecast
```

This produces filtered datasets such as:

- `processed_y2023_sacramento_event_q12h12`
- `processed_y2023_sacramento_noevent_q12h12`

## Notes on Current Engineering Choices

- The project now prioritizes Sacramento/XTraffic experiments over the original benchmark manuscript setup.
- Many current experiments are flow-only (`input_dim=1`), even when richer metadata or incident tensors exist.
- TimeCMA supports official-style external prompt embeddings and framework-native execution.
- E2CSTP supports multimodal channel partitioning via:

```text
input_dim = st_dim + text_dim + image_dim
```

- PatchTST is adapted to the native LargeST tensor format while preserving patch tokenization, Transformer encoding, and optional RevIN.

## Upstream Acknowledgement

This project is built on top of the original LargeST benchmark repository and retains substantial parts of its training framework, baseline organization, and data handling conventions.

If you are looking for the original benchmark paper and dataset release, please refer to:

- Paper: [LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting](https://arxiv.org/abs/2306.08259)

## Citation

If you use the original LargeST benchmark in your research, please cite:

```bibtex
@inproceedings{liu2023largest,
  title={LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting},
  author={Liu, Xu and Xia, Yutong and Liang, Yuxuan and Hu, Junfeng and Wang, Yiwei and Bai, Lei and Huang, Chao and Liu, Zhenguang and Hooi, Bryan and Zimmermann, Roger},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```
