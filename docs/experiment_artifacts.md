# Experiment Layout

Future experiments should use this layout:

```text
experiments/<model>/<run_dir>/
  record_s<seed>.log
  final_model_s<seed>.pt
  artifacts/
    launch/
    eval/
    embed/
    metrics/
    pid/
    waiter/
```

`run_dir` naming rule:

```text
ds-<dataset>__yr-<years>__q<seq_len>__h<horizon>__s<seed>__<extra-parts>__tag-<run_tag>
```

Examples:

```text
ds-SacraJan__yr-2023__q12__h12__s2023__flow-1__emb-64__tag-jan_flow_12to12
ds-Sacra__yr-2023__q24__h24__s2023__flow-1__dm-32__llm-llama32__prompt-stats__tag-sacra_llama32_24to24_gpu7_stride48
```

Notes:

- `record_s<seed>.log` remains the canonical in-run logger output.
- External `nohup`/evaluation/export logs should be redirected into `artifacts/<bucket>/`.
- Historical root-level logs can be normalized with `scripts/organize_experiment_artifacts.py`.
