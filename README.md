# The LargeST Benchmark Dataset

This is the official repository of our NeurIPS 2023 DB Track paper [LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting](https://arxiv.org/abs/2306.08259). LargeST comprises four sub-datasets, each characterized by a different number of sensors. The biggest one is California (CA), including a total number of 8,600 sensors. We also construct three subsets of CA by selecting three representative areas within CA and forming the sub-datasets of Greater Los Angeles (GLA), Greater Bay Area (GBA), and San Diego (SD). The figure here shows an illustration.

<img src='img/overview.png' width='780px'>

In LargeST we also provide comprehensive metadata for all sensors, which are listed below.
| Attribute |                 Description                     |  Possible Range of Values
|   :---    |                    :---                         |          :---
|    ID     |  The identifier of a sensor in PeMS             |  6 to 9 digits number
|    Lat    |  The latitude of a sensor                       |  Real number
|    Lng    |  The longitude of a sensor                      |  Real number
|  District |  The district of a sensor in PeMS               |  3, 4, 5, 6, 7, 8, 10, 11, 12
|   County  |  The county of a sensor in California           |  String
|    Fwy    |  The highway where a sensor is located          |  String starts with 'I', 'US', or 'SR'
|    Lane   |  The number of lanes where a sensor is located  |  1, 2, 3, 4, 5, 6, 7, 8
|    Type   |  The type of a sensor                           |  Mainline
| Direction |  The direction of the highway                   |  N, S, E, W


## 1. Data Preparation
In this section, we will outline the procedure for preparing the CA dataset, followed by an explanation of how the GLA, GBA, and SD datasets can be derived from CA. Please follow these instructions step by step.

### 1.1 Download the CA Dataset
We host the CA dataset on Kaggle: https://www.kaggle.com/datasets/liuxu77/largest. There are a total of 7 files in this link. Among them, 5 files in .h5 format contain the traffic flow raw data from 2017 to 2021, 1 file in .csv format provides the metadata for all sensors, and 1 file in .npy format represents the adjacency matrix constructed based on road network distances.

- **If you are using the web user interface**, you can download all data from the provided [link](https://www.kaggle.com/datasets/liuxu77/largest). The download button is at the upper right corner of the webpage. Then please place the downloaded archive.zip file in the `data/ca` folder and unzip the file.

- **If you would like to use the Kaggle API**, please follow the instructions [here](https://github.com/Kaggle/kaggle-api). After setting the API correctly, you can simply go to the `data/ca` folder, and use the command below to download all data.
```
kaggle datasets download liuxu77/largest
```

Note that the traffic flow raw data of the CA dataset require additional processing (described in Section 1.2 and 1.3), while the metadata and adjacency matrix are ready to be used.

### 1.2 Process Traffic Flow Data of CA
We provide a jupyter notebook `process_ca_his.ipynb` in the folder `data/ca` to process and generate a cleaned version of the flow data. Please go through this notebook.

### 1.3 Generate Traffic Flow Data for Training
Please go to the `data` folder, and use the command below to generate the flow data for model training in our manuscript.
```
python generate_data_for_training.py --dataset ca --years 2019
```
The processed data are stored in `data/ca/2019`. We also support the utilization of data from multiple years. For example, changing the years argument to 2018_2019 to generate two years of data.

### 1.4 Generate Other Sub-Datasets
We describe the generation of the GLA dataset as an example. Please first go through all the cells in the provided jupyter notebook `generate_gla_dataset.ipynb` in the folder `data/gla`. Then, use the command below to generate traffic flow data for model training.
```
python generate_data_for_training.py --dataset gla --years 2019
```


## 2. Experiments Running
We conduct experiments on an Intel(R) Xeon(R) Gold 6140 CPU @ 2.30 GHz, 376 GB RAM computing server, equipped with an NVIDIA RTX A6000 GPU with 48 GB memory. We adopt PyTorch 1.12 as the default deep learning library. Currently, there are a total of 19 supported baselines in this repository, namely, Historical Last (HL), LSTM, [DCRNN](https://github.com/chnsh/DCRNN_PyTorch), [AGCRN](https://github.com/LeiBAI/AGCRN), [STGCN](https://github.com/hazdzz/STGCN), [GWNET](https://github.com/nnzhan/Graph-WaveNet), [ASTGCN](https://github.com/guoshnBJTU/ASTGCN-r-pytorch), [STTN](https://github.com/xumingxingsjtu/STTN), [STGODE](https://github.com/square-coder/STGODE), [DSTAGNN](https://github.com/SYLan2019/DSTAGNN), [DGCRN](https://github.com/tsinghua-fib-lab/Traffic-Benchmark/tree/master/methods/DGCRN), [D2STGNN](https://github.com/zezhishao/D2STGNN), PatchTST, E2-CSTP, TimeCMA, Time-LLM, ST-LLM, CrossTrafficLLM, and STEVE.

To reproduce the benchmark results in the manuscript, please go to `experiments/baseline_you_want_to_run`, open the provided `run.sh` file, and uncomment the line you would like to execute. Note that you may need to specify the GPU card number on your server. Moreover, we use the flow data from 2019 for model training in our manuscript, if you want to use multiple years of data, please change the years argument to, e.g., 2018_2019.

To run the LSTM baseline, for example, you may execute this command in the terminal:
```
bash experiments/lstm/run.sh
```
or directly execute the Python file in the terminal:
```
python experiments/lstm/main.py --device cuda:2 --dataset SD --years 2019 --model_name lstm --seed 2023 --bs 64
```

To run the E2-CSTP reproduction in the original LargeST single-modal setting, you may execute:
```
python experiments/e2cstp/main.py --device cuda:0 --dataset SD --years 2019 --model_name e2cstp --seed 2023 --bs 64 --input_dim 3 --st_dim 3 --text_dim 0 --image_dim 0
```

For aligned multi-modal inputs, keep using the same `his.npz` file format and append the extra text/image channels to the feature dimension. Then set:
```
input_dim = st_dim + text_dim + image_dim
```
For example, if each timestamp-node pair has 3 traffic features, 8 text features, and 8 image features, use:
```
python experiments/e2cstp/main.py --device cuda:0 --dataset SD --years 2019 --model_name e2cstp --seed 2023 --bs 32 --input_dim 19 --st_dim 3 --text_dim 8 --image_dim 8
```
The current implementation keeps the repository dependency-free: the paper's causal adjacency estimation is approximated with an EMA-updated node-importance scorer over the prior road graph, while the cross-modal fusion, causal intervention branch, and GCN + Mamba-style spatio-temporal encoder are implemented directly.

To run the TimeCMA adaptation, you may execute:
```
python experiments/timecma/main.py --device cuda:0 --dataset SD --years 2019 --model_name timecma --input_dim 3 --ts_dim 3 --prompt_dim 0 --run_tag official_base
```
This TimeCMA integration now follows the official training recipe more closely: `AdamW`, `CosineAnnealingLR`, `masked_mse` training loss, `bs=16`, `channel=32`, `d_ff=32`, `dropout=0.2`, `patience=50`, and isolated experiment folders by `dataset/seq_len/horizon/run_tag`. To stay compatible with LargeST tensors, the first `ts` channel is treated as the main scalar series per node, which matches the original TimeCMA assumption.

To use prompt-aligned embeddings from the original [TimeCMA](https://github.com/ChenxiLiu-HNU/TimeCMA) setting, append them to the feature dimension of `his.npz`. Then set:
```
input_dim = ts_dim + prompt_dim
```
For example, with 3 traffic channels and 64 prompt channels:
```
python experiments/timecma/main.py --device cuda:0 --dataset SD --years 2019 --model_name timecma --input_dim 67 --ts_dim 3 --prompt_dim 64 --prompt_hidden 128 --prompt_pool mean --run_tag inline_prompt
```
This repository version keeps the official TimeCMA dual-branch layout and now supports two prompt-embedding modes:
- inline prompt channels appended to `his.npz` (as above), or
- external embeddings (official-style) passed to `forward` via dataloader.

To generate external prompt embeddings in the repository:
```
python scripts/generate_timecma_prompt_embeddings.py --data_path data/sd --years 2019 --ts_dim 3 --seq_len 12 --embedding_method gpt2 --d_llm 768 --external_prompt_dim 768 --data_name SD --start_datetime '2019-01-01 00:00:00' --freq_minutes 5 --device cuda:0
```
This script saves:
```
data/sd/2019/prompt_emb_train.npy
data/sd/2019/prompt_emb_val.npy
data/sd/2019/prompt_emb_test.npy
```
Then train TimeCMA with:
```
python experiments/timecma/main.py --device cuda:0 --dataset SD --years 2019 --model_name timecma --input_dim 3 --ts_dim 3 --prompt_dim 0 --use_external_embeddings 1 --embedding_prefix prompt_emb --d_llm 768 --external_prompt_dim 768 --prompt_data_name SD --prompt_start_datetime '2019-01-01 00:00:00' --prompt_freq_minutes 5 --run_tag official_external
```
For custom dataset folders, pass `--data_path` and (optionally) `--node_num`:
```
python experiments/timecma/main.py --device cuda:0 --dataset SD --data_path /path/to/your_dataset --node_num 517 --years 2023 --model_name timecma --input_dim 1 --ts_dim 1 --prompt_dim 0 --use_external_embeddings 1 --embedding_prefix prompt_emb --d_llm 768 --external_prompt_dim 768 --prompt_data_name Sacramento --prompt_start_datetime '2023-01-01 00:00:00' --prompt_freq_minutes 5 --run_tag sacra_external
```
You can also skip pre-saving and generate embeddings online during training:
```
python experiments/timecma/main.py --device cuda:0 --dataset SD --years 2019 --model_name timecma --input_dim 3 --ts_dim 3 --prompt_dim 0 --generate_embeddings_on_the_fly 1 --embedding_method gpt2 --d_llm 768 --external_prompt_dim 768 --prompt_data_name SD --prompt_start_datetime '2019-01-01 00:00:00' --prompt_freq_minutes 5 --run_tag official_online
```
Since TimeCMA uses node-wise attention and prompt generation can be expensive, it is best validated on smaller subsets such as SD first.

To run the PatchTST adaptation, you may execute:
```
python experiments/patchtst/main.py --device cuda:0 --dataset SD --years 2019 --model_name patchtst --seed 2023 --bs 64 --seq_len 12 --horizon 12 --input_dim 3 --traffic_dim 3
```
For custom Sacramento flow-only data, use:
```
python experiments/patchtst/main.py --device cuda:0 --dataset Sacra --years 2023 --model_name patchtst --seed 2023 --bs 16 --seq_len 12 --horizon 12 --input_dim 1 --traffic_dim 1 --data_path /root/XTraffic/data/processed_y2023_sacramento --node_num 517
```
This repository version keeps PatchTST's patch-based tokenizer, shared Transformer encoder, flatten prediction head, and optional RevIN normalization, while adapting input/output to the native LargeST tensor format `[B, T, N, F]`.

To run the Time-LLM integration, you may execute:
```
python experiments/timellm/main.py --device cuda:0 --dataset SD --years 2019 --model_name timellm --seed 2023 --bs 16 --seq_len 12 --horizon 12 --input_dim 3 --traffic_dim 3 --llm_model GPT2 --prompt_mode stats --prompt_granularity batch
```
For flow-only data (e.g., custom Sacramento subset), set `input_dim=traffic_dim=1` and pass a custom dataset root:
```
python experiments/timellm/main.py --device cuda:0 --dataset Sacra --years 2023 --model_name timellm --seed 2023 --bs 16 --seq_len 12 --horizon 12 --input_dim 1 --traffic_dim 1 --data_path /path/to/data_root --node_num 517 --llm_model GPT2 --prompt_mode stats --prompt_granularity batch
```
This repository version keeps Time-LLM's patch reprogramming and prompt-token conditioning idea, and supports loading GPT2/BERT/LLAMA backbones from HuggingFace (with local-files-first behavior configurable by runtime flags).

To enable SwanLab logging for training and testing metrics, append:
```
--use_swanlab 1 --swanlab_project LargeST --swanlab_experiment timellm_run_name
```
For heavy backbones on large node sets, it is recommended to sparsify training windows:
```
--node_chunk_size 8 --train_sample_stride 48 --val_sample_stride 12 --test_sample_stride 12 --log_interval 10
```
For Sacramento flow-only data with LLAMA 32 layers (12->12):
```
export HF_ENDPOINT=https://hf-mirror.com
python experiments/timellm/main.py --device cuda:0 --dataset Sacra --years 2023 --model_name timellm --seed 2023 --bs 1 --seq_len 12 --horizon 12 --input_dim 1 --traffic_dim 1 --data_path /root/XTraffic/data/processed_y2023_sacramento --node_num 517 --llm_model LLAMA --llm_layers 32 --llm_model_name huggyllama/llama-7b --llm_local_files_only 1 --llm_allow_download 1 --llm_torch_dtype float16 --prompt_mode stats --prompt_granularity batch --node_chunk_size 8 --train_sample_stride 48 --val_sample_stride 12 --test_sample_stride 12 --log_interval 10 --freeze_backbone 1 --use_swanlab 1 --swanlab_project LargeST --swanlab_experiment timellm_sacra_llama32_12to12
```

To run the ST-LLM adaptation (only ST_LLM model from [ST-LLM](https://github.com/ChenxiLiu-HNU/ST-LLM)), you may execute:
```
python experiments/stllm/main.py --device cuda:0 --dataset SD --years 2019 --model_name stllm --seed 2023 --bs 16 --seq_len 12 --horizon 12 --input_dim 3 --output_dim 1 --steps_per_day 288 --auto_time_features 1
```
This version supports automatic temporal feature generation in the dataloader: for each training window index, it dynamically appends `time_of_day` and `day_of_week` channels.

If your dataset is flow-only (e.g., only one traffic channel), keep:
```
--input_dim 1 --auto_time_features 1
```
This repository version keeps ST-LLM's spatial-temporal embedding design and partially frozen attention strategy over GPT-2, while adapting input/output to the native LargeST dataloader and training pipeline.

To run the CrossTrafficLLM adaptation for numeric traffic forecasting only, you may execute:
```
python experiments/crosstrafficllm/main.py --device cuda:0 --dataset SD --years 2019 --model_name crosstrafficllm --seed 2023 --bs 16 --input_dim 3 --traffic_dim 3 --text_dim 0
```

To use aligned text embeddings, append them to the feature dimension of `his.npz` and set:
```
input_dim = traffic_dim + text_dim
```
For example, with 3 traffic channels and 32 text embedding channels:
```
python experiments/crosstrafficllm/main.py --device cuda:0 --dataset SD --years 2019 --model_name crosstrafficllm --seed 2023 --bs 8 --input_dim 35 --traffic_dim 3 --text_dim 32 --hidden_dim 64 --text_hidden 64
```

The current CrossTrafficLLM adaptation keeps a text-guided adaptive graph encoder, sparse cross-modal alignment, a TextCrossformer-style prediction backbone, and an optional interpretable report-generation head. To train the report head, place sample-aligned integer token targets in:
```
data/<dataset>/<year>/report_train.npy
data/<dataset>/<year>/report_val.npy
data/<dataset>/<year>/report_test.npy
```
and set `--report_vocab_size > 0`. If these files are absent, the model automatically falls back to traffic prediction only. Since this version is adapted to the existing LargeST data pipeline, it should be treated as a repository-compatible implementation rather than an official reproduction of the original CrossTrafficLLM codebase.

To run the STEVE adaptation, you may execute:
```
python experiments/steve/main.py --device cuda:0 --dataset SD --years 2019 --model_name steve --seed 2023 --bs 16 --input_dim 3 --traffic_dim 3 --embed_dim 64
```
This repository version keeps STEVE's dual-branch robust prediction design and basis-bank confounder extractor, while adapting auxiliary supervision to LargeST's single-tensor dataloader.


## 3. Evaluate Your Model in Three Steps
You may first go through the implementations of various baselines in our repository, which may serve as good references for experimenting your own model. The detailed steps are described as follows.
- The first step is to define the model architecture, and place it into `src/models`. To ensure compatibility with the existing framework, it is recommended that your model inherits the BaseModel class (implemented in `src/base/model.py`).
- If your model does not require any special training or testing procedures beyond the standard workflow provided by the BaseEngine class (implemented in `src/base/engine.py`), you can directly use it for training and evaluation. Otherwise, please include a file in the folder `src/engines`.
- To integrate your model and engine files, you need to create a `main.py` file in the `experiments/your_model_name` directory.


## 4. License \& Acknowledgement
The LargeST benchmark dataset is released under a CC BY-NC 4.0 International License: https://creativecommons.org/licenses/by-nc/4.0. Our code implementation is released under the MIT License: https://opensource.org/licenses/MIT. The license of any specific baseline methods used in our codebase should be verified on their official repositories. Here we would also like to express our gratitude to the authors of baselines for releasing their code.


## 5. Citation
If you find our work useful in your research, please cite:
```
@inproceedings{liu2023largest,
  title={LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting},
  author={Liu, Xu and Xia, Yutong and Liang, Yuxuan and Hu, Junfeng and Wang, Yiwei and Bai, Lei and Huang, Chao and Liu, Zhenguang and Hooi, Bryan and Zimmermann, Roger},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}

```
