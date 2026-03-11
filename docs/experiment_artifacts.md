# Experiment Layout

Future experiments should use this layout:

```text
experiments/<model>/<run_dir>/
  record_s<seed>.log
  final_model_s<seed>.pt
  launch.log
  launch.pid
  run.sh
  eval.log
  metrics.json
```

`run_dir` naming rule:

```text
<model>_<dataset>_q<seq_len>_h<horizon>_s<seed>_t<YYMMDDHHMM>
```

Examples:

```text
steve_SacraJan2023_q12_h12_s2023_t2603090329
timecma_SacraJan2023_q24_h24_s2024_t2603091230
```

Notes:

- `record_s<seed>.log` remains the canonical in-run logger output.
- External `nohup`/pipeline/evaluation logs should be redirected into the run directory root.
- Use `--desc "..."` or `--swanlab_description "..."` to mark experiment purpose in SwanLab.
- Historical root-level logs can be normalized with `scripts/organize_experiment_artifacts.py`.
