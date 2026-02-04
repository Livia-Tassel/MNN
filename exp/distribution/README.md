实验目标：在写入 KV Cache 之前导出 K/V 张量（逻辑布局），分析各层分布与压缩率，并验证 RoPE 是否降低压缩率。

**编译开关**
- `-DMNN_BUILD_LLM=ON`
- `-DMNN_SUPPORT_TRANSFORMER_FUSE=ON`
- `-DMNN_EXP_KV_DUMP=ON`

**运行时环境变量**
- `MNN_KV_DUMP_DIR`：dump 输出目录，默认 `exp/distribution/dumps`
- `MNN_KV_DUMP_RUN_ID`：可选，指定 run 目录名
- `MNN_KV_DUMP_MAX_TOKENS`：单次 dump 最多 token 数，默认不限制
- `MNN_KV_DUMP_STAGE`：`prefill` / `decode` / `both`，默认 `prefill`

**生成 KV dump（示例）**
```bash
export MNN_KV_DUMP_DIR=exp/distribution/dumps
export MNN_KV_DUMP_STAGE=prefill
./path/to/llm_bench -m /path/to/model/config.json -kv true -p 128 -n 0
./path/to/llm_bench -m /path/to/model/config.json -kv true -p 512 -n 0
./path/to/llm_bench -m /path/to/model/config.json -kv true -p 2048 -n 0
```

**分析分布与压缩率**
```bash
python3 exp/distribution/analyze_distribution.py \
  --dump-dir exp/distribution/dumps \
  --out-dir exp/distribution/out \
  --zstd-level 3 \
  --stage prefill
```

**可选：RoPE 逆变换对比**
```bash
python3 exp/distribution/analyze_distribution.py \
  --dump-dir exp/distribution/dumps \
  --out-dir exp/distribution/out \
  --zstd-level 3 \
  --stage prefill \
  --rope-config /path/to/llm_config.json
```

**输出**
- `exp/distribution/out/layer_metrics.csv`：逐层统计结果
- `exp/distribution/out/summary.md`：总体平均值汇总
