# RL data pipeline

This directory contains the minimal data-building pipeline used by geometric
query training and Query-Reflect GRPO. Generated datasets are intentionally
not tracked.

`build_stage1_data.py` can start either from a frozen base MLLM plus SAM2 or
from an existing geometric export. The default launcher uses the base-MLLM
mode; set `STAGE1_EXPORT_DIR` to use an export instead. The generated stage-1
cache is data preprocessing, not an additional model-training stage.

## Data provenance

| Generated file | Generator | Main inputs |
|---|---|---|
| `query_reflect_grpo_stage1/init_box_train.json` | `build_stage1_data.py` | Training annotations, images, base MLLM (or export), SAM2 |
| `query_reflect_grpo_stage1/init_box_val.json` | `build_stage1_data.py` | Validation annotations, images, base MLLM (or export), SAM2 |
| `query_reflect_grpo_stage1/stage1_cache_train.json` | `build_stage1_data.py` | Training annotations, images, base MLLM (or export), SAM2 |
| `query_reflect_grpo_stage1/stage1_cache_val.json` | `build_stage1_data.py` | Validation annotations, images, base MLLM (or export), SAM2 |
| `query_reflect_grpo_reflect/reflect_train.json` | `build_reflect_samples.py` | `stage1_cache_train.json`, reflection MLLM |
| `query_reflect_grpo_reflect/reflect_val.json` | `build_reflect_samples.py` | `stage1_cache_val.json`, reflection MLLM |
| `query_reflect_grpo/train.json` | `build_unified_grpo_data.py` | init-box records, reflection records, stage-1 cache |
| `query_reflect_grpo/val.json` | `build_unified_grpo_data.py` | init-box records, reflection records, stage-1 cache |
| `query_reflect_grpo_balanced/train.json` | `balance_unified_grpo_data.py` | Unified `train.json` |
| `query_reflect_grpo_balanced/val.json` | `balance_unified_grpo_data.py` | Unified `val.json` |

The corresponding launchers are:

1. `run_build_stage1_data_4x80G.sh`
2. `run_build_reflect_samples_4x80G.sh`
3. `run_build_unified_grpo_data_4x80G.sh`
4. `run_balance_unified_grpo_data.sh` (optional)

Each launcher accepts paths through environment variables. See the variable
definitions at the top of the launcher before running it.
