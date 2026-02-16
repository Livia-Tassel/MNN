# H2O v6 Runtime + Real Prompt Gate Kit

v6 goals:
- Keep v5 three-target quality level (lossy >= 3.0, online-sim lossless >= 1.3)
- Add real-prompt validation through `llm_demo`
- Land async encode (single in-flight worker) + decode-cache reuse observability

## Default Paths (server-friendly)
- `MODEL_CONFIG=/home10T/ljq/mnn_data/models/llama2_mnn/config.json`
- `PROMPT_DIR=/home10T/ljq/MNN/exp/gear_fp16/prompts`
- `LLM_BENCH=./build/llm_bench`
- `LLM_DEMO=./build/llm_demo`

## Core Scripts
- `run_h2o_v6_bench.py`: generate configs + run `llm_bench`
- `sweep_h2o_v6.py`: run preset sweeps
- `parse_h2o_v6_log.py`: parse markdown logs to CSV
- `analyze_h2o_v6.py`: gate summary (supports async/decode-cache gates)
- `test_v6_runtime.sh`: strict runtime gate
- `test_v6_m3.sh`: multi-run stability pack
- `run_llm_demo_real_prompt.py`: batch run real prompts via `llm_demo`
- `test_v6_llm_demo.sh`: baseline vs candidate real-prompt gate
- `test_v6_final.sh`: runtime + m3 + llm_demo final acceptance

## Recommended Runs

### 1) Runtime strict gate
```bash
bash exp/h2o_v6/test_v6_runtime.sh
```

### 2) M3 pack
```bash
bash exp/h2o_v6/test_v6_m3.sh
```

### 3) Real prompt gate (`llm_demo`)
```bash
bash exp/h2o_v6/test_v6_llm_demo.sh
```

### 4) One-command final acceptance
```bash
bash exp/h2o_v6/test_v6_final.sh
```

## Optional Strictness Knobs
- `MAX_LOSSLESS_DECOMP_US`: runtime decomp budget
- `MAX_LOSSLESS_ASYNC_WAIT_US`: async wait budget
- `STRICT_RUNTIME_METRIC_COLUMNS=1`: fail if runtime metric columns are missing in CSV
- `REQUIRE_DECODE_CACHE_HIT=1`: require decode cache hit evidence
- `REQUIRE_ASYNC_QUEUE_ACTIVITY=1`: require async queue peak > 0 evidence
- `REQUIRE_DECODE_CACHE_ACTIVITY=1`: require decode cache (hit+miss) activity > 0
- `KV_LOSSLESS_ASYNC_THREADS`: async encode threads (v6 first release targets `1`)
- `KV_LOSSLESS_DECODE_CACHE_BLOCKS`: decode cache capacity

## Real Prompt Sampling Knobs (`test_v6_llm_demo.sh`)
- `PROMPT_PATTERN='prompt_???_*.txt'`: glob filter (e.g. run only 128/512 prompts)
- `PROMPT_MANIFEST=/path/to/prompts_manifest.jsonl`: use explicit prompt file list
- `MAX_PROMPTS=6`: hard cap selected prompts for quick iteration

## Presets
- `configs/target_core_gate_v6.json`
- `configs/target_joint_scope_v6.json`
- `configs/target_joint_scope_v6_aggressive.json`
- `configs/target_joint_scope_v6_full_coverage.json`
- `configs/target_store_v6.json`
- `configs/target_ratio_v6.json`
- `configs/target_speed_v6.json`
