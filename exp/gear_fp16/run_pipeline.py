#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def run_cmd(cmd, cwd=None):
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end pipeline: llm_demo -> FP16 convert -> analysis.")
    parser.add_argument("--llm-demo", default="./build/llm_demo", help="Path to llm_demo binary.")
    parser.add_argument("--config", required=True, help="Path to model config.json.")
    parser.add_argument("--prompt-dir", default="exp/gear_fp16/prompts", help="Directory containing prompt .txt files.")
    parser.add_argument("--prompt-files", nargs="*", default=[], help="Prompt files list.")
    parser.add_argument("--corpus", nargs="*", default=[], help="Corpus file(s) or directories for prompt generation.")
    parser.add_argument("--gen-prompts", action="store_true", help="Generate prompts before running llm_demo.")
    parser.add_argument("--prompt-targets", default="128,512,2048", help="Comma-separated target token lengths.")
    parser.add_argument("--prompt-per-target", type=int, default=3)
    parser.add_argument("--prompt-tokenizer", choices=["auto", "mnn", "hf", "approx"], default="auto")
    parser.add_argument("--prompt-hf-path", default=None)
    parser.add_argument("--prompt-tolerance", type=float, default=0.05)
    parser.add_argument("--prompt-min-piece-chars", type=int, default=200)
    parser.add_argument("--prompt-strict", action="store_true")
    parser.add_argument("--prompt-overwrite", action="store_true")
    parser.add_argument("--decode-tokens", type=int, default=128, help="Max new tokens for decode.")
    parser.add_argument("--dump-dir", default="exp/gear_fp16/dumps_raw", help="KV dump output root dir.")
    parser.add_argument("--fp16-dump-dir", default="exp/gear_fp16/dumps_fp16", help="FP16 dump output dir.")
    parser.add_argument("--out-dir", default="exp/gear_fp16/out", help="Analysis output dir.")
    parser.add_argument("--stage", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--max-dump-tokens", type=int, default=0, help="Limit dump seq len (0 = no limit).")
    parser.add_argument("--zstd-level", type=int, default=3, help="Zstd compression level.")
    parser.add_argument("--min-seq-len", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=0)
    parser.add_argument("--skip-head-metrics", action="store_true")
    parser.add_argument("--run-prefix", default="demo", help="Prefix for MNN_KV_DUMP_RUN_ID.")
    parser.add_argument("--log-file", default="exp/gear_fp16/out/llm_demo_runs.jsonl")
    parser.add_argument("--skip-demo", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    gen_prompts = os.path.join(root, "gen_prompts.py")
    run_demo = os.path.join(root, "run_llm_demo_batch.py")
    convert = os.path.join(root, "convert_fp16.py")
    analyze = os.path.join(root, "analyze_fp16.py")

    if args.gen_prompts or args.corpus:
        cmd = [
            sys.executable, gen_prompts,
            "--config", args.config,
            "--out-dir", args.prompt_dir,
            "--targets", args.prompt_targets,
            "--per-target", str(args.prompt_per_target),
            "--tokenizer", args.prompt_tokenizer,
            "--tolerance", str(args.prompt_tolerance),
            "--min-piece-chars", str(args.prompt_min_piece_chars),
        ]
        if args.corpus:
            cmd += ["--corpus"] + args.corpus
        if args.prompt_hf_path:
            cmd += ["--hf-path", args.prompt_hf_path]
        if args.prompt_strict:
            cmd.append("--strict")
        if args.prompt_overwrite:
            cmd.append("--overwrite")
        run_cmd(cmd)

    if not args.skip_demo:
        cmd = [
            sys.executable, run_demo,
            "--llm-demo", args.llm_demo,
            "--config", args.config,
            "--dump-dir", args.dump_dir,
            "--run-prefix", args.run_prefix,
            "--stage", args.stage,
            "--decode-tokens", str(args.decode_tokens),
            "--log-file", args.log_file,
        ]
        if args.prompt_dir:
            cmd += ["--prompt-dir", args.prompt_dir]
        if args.prompt_files:
            cmd += ["--prompt-files"] + args.prompt_files
        if args.max_dump_tokens > 0:
            cmd += ["--max-dump-tokens", str(args.max_dump_tokens)]
        run_cmd(cmd)

    if not args.skip_convert:
        cmd = [
            sys.executable, convert,
            "--src-dump-dir", args.dump_dir,
            "--dst-dump-dir", args.fp16_dump_dir,
            "--stage", args.stage,
            "--min-seq-len", str(args.min_seq_len),
            "--max-seq-len", str(args.max_seq_len),
        ]
        run_cmd(cmd)

    if not args.skip_analyze:
        cmd = [
            sys.executable, analyze,
            "--dump-dir", args.fp16_dump_dir,
            "--out-dir", args.out_dir,
            "--stage", args.stage,
            "--zstd-level", str(args.zstd_level),
            "--min-seq-len", str(args.min_seq_len),
            "--max-seq-len", str(args.max_seq_len),
        ]
        if args.log_file:
            cmd += ["--run-log", args.log_file]
        if args.skip_head_metrics:
            cmd.append("--skip-head-metrics")
        run_cmd(cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
