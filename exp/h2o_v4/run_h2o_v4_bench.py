#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import subprocess
from pathlib import Path


def parse_list(text, cast):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def run_cmd(cmd):
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def normalize_base_dir(base_cfg, base_config_path):
    model_cfg_dir = Path(base_config_path).resolve().parent
    raw_base_dir = base_cfg.get("base_dir", "")
    if isinstance(raw_base_dir, str) and raw_base_dir:
        base_dir_path = Path(raw_base_dir)
        if not base_dir_path.is_absolute():
            base_dir_path = (model_cfg_dir / base_dir_path).resolve()
        else:
            base_dir_path = base_dir_path.resolve()
    else:
        base_dir_path = model_cfg_dir
    base_dir = os.fspath(base_dir_path)
    if not base_dir.endswith(("/", "\\")):
        base_dir += "/"
    return base_dir


def normalize_mnn_path_field(cfg, key, base_dir, default_value):
    raw = cfg.get(key, default_value)
    if not isinstance(raw, str) or not raw:
        return
    base_dir_path = Path(base_dir).resolve()
    p = Path(raw)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (base_dir_path / p).resolve()
    # LlmConfig always concatenates `base_dir + field_value`, so field value must be relative.
    rel = os.path.relpath(os.fspath(resolved), os.fspath(base_dir_path))
    cfg[key] = rel.replace("\\", "/")


def main():
    parser = argparse.ArgumentParser(description="Run llm_bench sweeps for H2O v4.")
    parser.add_argument("--llm-bench", required=True, help="Path to llm_bench binary.")
    parser.add_argument("--base-config", required=True, help="Base model config json.")
    parser.add_argument("--out-dir", default="exp/h2o_v4/out")
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--attention-option", type=int, default=0)
    parser.add_argument("--prompts", default="512")
    parser.add_argument("--gens", default="128")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--h2o-keep-ratios", default="0.5")
    parser.add_argument("--h2o-block-tokens", default="64")
    parser.add_argument("--h2o-sink-tokens", default="32")
    parser.add_argument("--h2o-recent-tokens", default="256")
    parser.add_argument("--h2o-ema-alphas", default="0.9")
    parser.add_argument("--h2o-update-intervals", default="16")
    parser.add_argument("--h2o-trigger-min-tokens", default="512")
    parser.add_argument("--h2o-layer-start", type=int, default=2)
    parser.add_argument("--h2o-layer-end", type=int, default=-1)
    parser.add_argument("--h2o-target-mode", default="adaptive", choices=["static", "adaptive"])
    parser.add_argument("--h2o-target-lossy-ratios", default="3.0")
    parser.add_argument("--h2o-log-stats", action="store_true")
    parser.add_argument("--kv-lossless-enable", action="store_true")
    parser.add_argument(
        "--kv-lossless-scope",
        default="front_n",
        choices=["none", "front_n", "h2o_kept", "front_n_and_h2o_kept", "front_n+h2o_kept"],
    )
    parser.add_argument("--kv-lossless-front-n", type=int, default=2)
    parser.add_argument("--kv-lossless-codec", default="none", choices=["none", "gear_delta"])
    parser.add_argument("--kv-lossless-runtime-enable", action="store_true")
    parser.add_argument("--kv-lossless-runtime-mode", default="probe", choices=["probe", "full", "store"])
    parser.add_argument("--kv-lossless-block-tokens", type=int, default=128)
    parser.add_argument("--kv-lossless-hot-recent-tokens", type=int, default=256)
    parser.add_argument("--kv-lossless-hot-sink-tokens", type=int, default=16)
    parser.add_argument("--kv-lossless-kept-sample-layers", type=int, default=1)
    parser.add_argument("--kv-lossless-kept-sample-token-interval", type=int, default=1)
    parser.add_argument("--kv-lossless-codec-runtime", default="fp16_gear_predictive_v3")
    parser.add_argument("--kv-lossless-predictors-k", default="raw,delta_seq,xor_seq,pair_delta")
    parser.add_argument("--kv-lossless-predictors-v", default="raw,delta_seq,xor_seq")
    parser.add_argument("--kv-lossless-async-threads", type=int, default=1)
    parser.add_argument("--kv-lossless-max-queue", type=int, default=256)
    parser.add_argument("--kv-lossless-decode-cache-blocks", type=int, default=64)
    parser.add_argument("--kv-lossless-strict-roundtrip-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cfg_dir = out_dir / "configs"
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    normalized_base_dir = normalize_base_dir(base_cfg, args.base_config)
    prompts = parse_list(args.prompts, int)
    gens = parse_list(args.gens, int)
    keep_ratios = parse_list(args.h2o_keep_ratios, float)
    block_tokens = parse_list(args.h2o_block_tokens, int)
    sink_tokens = parse_list(args.h2o_sink_tokens, int)
    recent_tokens = parse_list(args.h2o_recent_tokens, int)
    ema_alphas = parse_list(args.h2o_ema_alphas, float)
    update_intervals = parse_list(args.h2o_update_intervals, int)
    trigger_mins = parse_list(args.h2o_trigger_min_tokens, int)
    target_lossy_ratios = parse_list(args.h2o_target_lossy_ratios, float)

    sweep_dims = {
        "prompts": prompts,
        "gens": gens,
        "keep_ratios": keep_ratios,
        "block_tokens": block_tokens,
        "sink_tokens": sink_tokens,
        "recent_tokens": recent_tokens,
        "ema_alphas": ema_alphas,
        "update_intervals": update_intervals,
        "trigger_mins": trigger_mins,
        "target_lossy_ratios": target_lossy_ratios,
    }
    empty = [name for name, vals in sweep_dims.items() if not vals]
    if empty:
        raise SystemExit(f"Empty sweep dimension(s): {', '.join(empty)}. "
                         f"Check argument values — 0 runs would be produced.")

    manifest = []
    any_fail = False
    run_idx = 0
    for combo in itertools.product(
        prompts,
        gens,
        keep_ratios,
        block_tokens,
        sink_tokens,
        recent_tokens,
        ema_alphas,
        update_intervals,
        trigger_mins,
        target_lossy_ratios,
    ):
        prompt_len, gen_len, keep, block, sink, recent, alpha, update_int, trigger_min, target_lossy = combo
        run_idx += 1
        run_id = f"run_{run_idx:04d}_p{prompt_len}_g{gen_len}_k{keep}_b{block}_lr{target_lossy}"
        cfg_path = cfg_dir / f"{run_id}.json"
        log_path = log_dir / f"{run_id}.log"

        cfg = dict(base_cfg)
        cfg.update(
            {
                "base_dir": normalized_base_dir,
                "kv_h2o_enable": True,
                "kv_h2o_layer_start": int(args.h2o_layer_start),
                "kv_h2o_layer_end": int(args.h2o_layer_end),
                "kv_h2o_block_tokens": int(block),
                "kv_h2o_sink_tokens": int(sink),
                "kv_h2o_recent_tokens": int(recent),
                "kv_h2o_target_keep_ratio": float(keep),
                "kv_h2o_target_mode": args.h2o_target_mode,
                "kv_h2o_target_lossy_ratio": float(target_lossy),
                "kv_h2o_ema_alpha": float(alpha),
                "kv_h2o_update_interval": int(update_int),
                "kv_h2o_trigger_min_tokens": int(trigger_min),
                "kv_h2o_log_stats": bool(args.h2o_log_stats),
                "kv_lossless_enable": bool(args.kv_lossless_enable),
                "kv_lossless_scope": args.kv_lossless_scope,
                "kv_lossless_front_n": int(args.kv_lossless_front_n),
                "kv_lossless_codec": args.kv_lossless_codec,
                "kv_lossless_runtime_enable": bool(args.kv_lossless_runtime_enable),
                "kv_lossless_runtime_mode": args.kv_lossless_runtime_mode,
                "kv_lossless_block_tokens": int(args.kv_lossless_block_tokens),
                "kv_lossless_hot_recent_tokens": int(args.kv_lossless_hot_recent_tokens),
                "kv_lossless_hot_sink_tokens": int(args.kv_lossless_hot_sink_tokens),
                "kv_lossless_kept_sample_layers": int(args.kv_lossless_kept_sample_layers),
                "kv_lossless_kept_sample_token_interval": int(args.kv_lossless_kept_sample_token_interval),
                "kv_lossless_codec_runtime": args.kv_lossless_codec_runtime,
                "kv_lossless_predictors_k": args.kv_lossless_predictors_k,
                "kv_lossless_predictors_v": args.kv_lossless_predictors_v,
                "kv_lossless_async_threads": int(args.kv_lossless_async_threads),
                "kv_lossless_max_queue": int(args.kv_lossless_max_queue),
                "kv_lossless_decode_cache_blocks": int(args.kv_lossless_decode_cache_blocks),
                "kv_lossless_strict_roundtrip_check": bool(args.kv_lossless_strict_roundtrip_check),
            }
        )
        # Normalize model resource fields for MNN style path join:
        # final path = base_dir + field_value
        normalize_mnn_path_field(cfg, "llm_config", normalized_base_dir, "llm_config.json")
        normalize_mnn_path_field(cfg, "llm_model", normalized_base_dir, "llm.mnn")
        normalize_mnn_path_field(cfg, "llm_weight", normalized_base_dir, "llm.mnn.weight")
        normalize_mnn_path_field(cfg, "embedding_file", normalized_base_dir, "embeddings_bf16.bin")
        normalize_mnn_path_field(cfg, "tokenizer_file", normalized_base_dir, "tokenizer.txt")
        normalize_mnn_path_field(cfg, "context_file", normalized_base_dir, "context.json")
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=True, indent=2), encoding="utf-8")

        cmd = [
            args.llm_bench,
            "-m",
            str(cfg_path),
            "-a",
            args.backend,
            "-t",
            str(args.threads),
            "-qatten",
            str(args.attention_option),
            "-kv",
            "true",
            "-p",
            str(prompt_len),
            "-n",
            str(gen_len),
            "-rep",
            str(args.repeat),
            "-fp",
            str(log_path),
        ]

        rc = 0
        if not args.dry_run:
            rc = run_cmd(cmd)
            if rc != 0:
                any_fail = True
                print(f"[WARN] {run_id} failed with return code {rc}")
        manifest.append(
            {
                "run_id": run_id,
                "prompt_len": prompt_len,
                "gen_len": gen_len,
                "keep_ratio": keep,
                "block_tokens": block,
                "sink_tokens": sink,
                "recent_tokens": recent,
                "ema_alpha": alpha,
                "update_interval": update_int,
                "trigger_min_tokens": trigger_min,
                "h2o_layer_start": int(args.h2o_layer_start),
                "h2o_layer_end": int(args.h2o_layer_end),
                "h2o_target_mode": args.h2o_target_mode,
                "h2o_target_lossy_ratio": target_lossy,
                "kv_lossless_enable": bool(args.kv_lossless_enable),
                "kv_lossless_scope": args.kv_lossless_scope,
                "kv_lossless_front_n": int(args.kv_lossless_front_n),
                "kv_lossless_codec": args.kv_lossless_codec,
                "kv_lossless_runtime_enable": bool(args.kv_lossless_runtime_enable),
                "kv_lossless_runtime_mode": args.kv_lossless_runtime_mode,
                "kv_lossless_block_tokens": int(args.kv_lossless_block_tokens),
                "kv_lossless_hot_recent_tokens": int(args.kv_lossless_hot_recent_tokens),
                "kv_lossless_hot_sink_tokens": int(args.kv_lossless_hot_sink_tokens),
                "kv_lossless_kept_sample_layers": int(args.kv_lossless_kept_sample_layers),
                "kv_lossless_kept_sample_token_interval": int(args.kv_lossless_kept_sample_token_interval),
                "kv_lossless_codec_runtime": args.kv_lossless_codec_runtime,
                "kv_lossless_predictors_k": args.kv_lossless_predictors_k,
                "kv_lossless_predictors_v": args.kv_lossless_predictors_v,
                "kv_lossless_async_threads": int(args.kv_lossless_async_threads),
                "kv_lossless_max_queue": int(args.kv_lossless_max_queue),
                "kv_lossless_decode_cache_blocks": int(args.kv_lossless_decode_cache_blocks),
                "kv_lossless_strict_roundtrip_check": bool(args.kv_lossless_strict_roundtrip_check),
                "config": str(cfg_path),
                "log": str(log_path),
                "return_code": rc,
            }
        )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {out_dir / 'manifest.json'}")
    if any_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
