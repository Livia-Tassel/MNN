#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


def list_to_arg(values):
    return ",".join(str(v) for v in values)


def main():
    parser = argparse.ArgumentParser(description="Run preset H2O sweep configs.")
    parser.add_argument("--llm-bench", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--preset", required=True, help="Preset json, see exp/h2o_v2/configs/*.json")
    parser.add_argument("--out-dir", default="exp/h2o_v2/out")
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--attention-option", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    preset = json.loads(Path(args.preset).read_text(encoding="utf-8"))
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("run_h2o_v2_bench.py")),
        "--llm-bench",
        args.llm_bench,
        "--base-config",
        args.base_config,
        "--out-dir",
        args.out_dir,
        "--backend",
        args.backend,
        "--threads",
        str(args.threads),
        "--attention-option",
        str(args.attention_option),
        "--prompts",
        list_to_arg(preset.get("prompts", [512])),
        "--gens",
        list_to_arg(preset.get("gens", [128])),
        "--repeat",
        str(preset.get("repeat", 3)),
        "--h2o-keep-ratios",
        list_to_arg(preset.get("h2o_keep_ratios", [0.5])),
        "--h2o-block-tokens",
        list_to_arg(preset.get("h2o_block_tokens", [64])),
        "--h2o-sink-tokens",
        list_to_arg(preset.get("h2o_sink_tokens", [32])),
        "--h2o-recent-tokens",
        list_to_arg(preset.get("h2o_recent_tokens", [256])),
        "--h2o-ema-alphas",
        list_to_arg(preset.get("h2o_ema_alphas", [0.9])),
        "--h2o-update-intervals",
        list_to_arg(preset.get("h2o_update_intervals", [16])),
        "--h2o-trigger-min-tokens",
        list_to_arg(preset.get("h2o_trigger_min_tokens", [512])),
        "--h2o-layer-start",
        str(preset.get("h2o_layer_start", 2)),
        "--h2o-layer-end",
        str(preset.get("h2o_layer_end", -1)),
        "--h2o-target-mode",
        str(preset.get("h2o_target_mode", "adaptive")),
        "--h2o-target-lossy-ratios",
        list_to_arg(preset.get("h2o_target_lossy_ratios", [preset.get("h2o_target_lossy_ratio", 3.0)])),
        "--kv-lossless-scope",
        str(preset.get("kv_lossless_scope", "front_n")),
        "--kv-lossless-front-n",
        str(preset.get("kv_lossless_front_n", 2)),
    ]
    if preset.get("kv_lossless_enable", False):
        cmd.append("--kv-lossless-enable")
    codec = preset.get("kv_lossless_codec", "none")
    cmd.extend(["--kv-lossless-codec", str(codec)])
    if preset.get("h2o_log_stats", False):
        cmd.append("--h2o-log-stats")
    if args.dry_run:
        cmd.append("--dry-run")

    print("Running:", " ".join(cmd))
    rc = subprocess.run(cmd, check=False).returncode
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
