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


def main():
    parser = argparse.ArgumentParser(description="Run llm_bench sweeps for H2O v1.")
    parser.add_argument("--llm-bench", required=True, help="Path to llm_bench binary.")
    parser.add_argument("--base-config", required=True, help="Base model config json.")
    parser.add_argument("--out-dir", default="exp/h2o_v1/out")
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
    parser.add_argument("--h2o-log-stats", action="store_true")
    parser.add_argument("--kv-lossless-enable", action="store_true")
    parser.add_argument("--kv-lossless-codec", default="none", choices=["none", "gear_delta"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cfg_dir = out_dir / "configs"
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    prompts = parse_list(args.prompts, int)
    gens = parse_list(args.gens, int)
    keep_ratios = parse_list(args.h2o_keep_ratios, float)
    block_tokens = parse_list(args.h2o_block_tokens, int)
    sink_tokens = parse_list(args.h2o_sink_tokens, int)
    recent_tokens = parse_list(args.h2o_recent_tokens, int)
    ema_alphas = parse_list(args.h2o_ema_alphas, float)
    update_intervals = parse_list(args.h2o_update_intervals, int)
    trigger_mins = parse_list(args.h2o_trigger_min_tokens, int)

    manifest = []
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
    ):
        prompt_len, gen_len, keep, block, sink, recent, alpha, update_int, trigger_min = combo
        run_idx += 1
        run_id = f"run_{run_idx:04d}_p{prompt_len}_g{gen_len}_k{keep}_b{block}"
        cfg_path = cfg_dir / f"{run_id}.json"
        log_path = log_dir / f"{run_id}.log"

        cfg = dict(base_cfg)
        cfg.update(
            {
                "kv_h2o_enable": True,
                "kv_h2o_block_tokens": int(block),
                "kv_h2o_sink_tokens": int(sink),
                "kv_h2o_recent_tokens": int(recent),
                "kv_h2o_target_keep_ratio": float(keep),
                "kv_h2o_ema_alpha": float(alpha),
                "kv_h2o_update_interval": int(update_int),
                "kv_h2o_trigger_min_tokens": int(trigger_min),
                "kv_h2o_log_stats": bool(args.h2o_log_stats),
                "kv_lossless_enable": bool(args.kv_lossless_enable),
                "kv_lossless_codec": args.kv_lossless_codec,
            }
        )
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
                "config": str(cfg_path),
                "log": str(log_path),
                "return_code": rc,
            }
        )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()

