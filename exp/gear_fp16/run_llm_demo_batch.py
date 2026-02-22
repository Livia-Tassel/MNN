#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from typing import Dict, List


PROMPT_TOKENS_RE = re.compile(r"prompt tokens num = (\\d+)")
DECODE_TOKENS_RE = re.compile(r"decode tokens num = (\\d+)")
PREFILL_TIME_RE = re.compile(r"prefill time = ([\\d\\.]+) s")
DECODE_TIME_RE = re.compile(r"decode time = ([\\d\\.]+) s")


def collect_prompt_files(prompt_dir: str, prompt_files: List[str]) -> List[str]:
    files = list(prompt_files)
    if prompt_dir:
        for name in sorted(os.listdir(prompt_dir)):
            if name.endswith(".txt"):
                files.append(os.path.join(prompt_dir, name))
    return files


def parse_metrics(output: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    m = PROMPT_TOKENS_RE.search(output)
    if m:
        metrics["prompt_tokens"] = int(m.group(1))
    m = DECODE_TOKENS_RE.search(output)
    if m:
        metrics["decode_tokens"] = int(m.group(1))
    m = PREFILL_TIME_RE.search(output)
    if m:
        metrics["prefill_s"] = float(m.group(1))
    m = DECODE_TIME_RE.search(output)
    if m:
        metrics["decode_s"] = float(m.group(1))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run llm_demo on a batch of prompt files and dump KV caches.")
    parser.add_argument("--llm-demo", default="./build/llm_demo", help="Path to llm_demo binary.")
    parser.add_argument("--config", required=True, help="Path to model config.json.")
    parser.add_argument("--prompt-dir", default="", help="Directory containing prompt .txt files.")
    parser.add_argument("--prompt-files", nargs="*", default=[], help="Prompt files list.")
    parser.add_argument("--decode-tokens", type=int, default=128, help="Max new tokens for decode.")
    parser.add_argument("--dump-dir", default="exp/gear_fp16/dumps_raw", help="KV dump output root dir.")
    parser.add_argument("--run-prefix", default="demo", help="Prefix for MNN_KV_DUMP_RUN_ID.")
    parser.add_argument("--stage", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--max-dump-tokens", type=int, default=0, help="Limit dump seq len (0 = no limit).")
    parser.add_argument("--log-file", default="exp/gear_fp16/out/llm_demo_runs.jsonl", help="Log file path.")
    parser.add_argument("--extra-env", action="append", default=[], help="Extra env vars KEY=VALUE.")
    args = parser.parse_args()

    prompt_files = collect_prompt_files(args.prompt_dir, args.prompt_files)
    if not prompt_files:
        print("No prompt files found. Provide --prompt-dir or --prompt-files.", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    extra_env = {}
    for item in args.extra_env:
        if "=" not in item:
            print(f"Invalid --extra-env value: {item}", file=sys.stderr)
            return 1
        k, v = item.split("=", 1)
        extra_env[k] = v

    results = []
    for prompt_path in prompt_files:
        prompt_name = os.path.splitext(os.path.basename(prompt_path))[0]
        run_id = f"{args.run_prefix}_{prompt_name}"

        env = os.environ.copy()
        env.update(extra_env)
        env["MNN_KV_DUMP_DIR"] = args.dump_dir
        env["MNN_KV_DUMP_RUN_ID"] = run_id
        env["MNN_KV_DUMP_STAGE"] = args.stage
        if args.max_dump_tokens > 0:
            env["MNN_KV_DUMP_MAX_TOKENS"] = str(args.max_dump_tokens)

        cmd = [args.llm_demo, args.config, prompt_path, str(args.decode_tokens)]
        print("Running:", " ".join(cmd))
        proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        metrics = parse_metrics(output)

        result = {
            "prompt_file": prompt_path,
            "run_id": run_id,
            "returncode": proc.returncode,
            "metrics": metrics,
        }
        results.append(result)

        with open(args.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if proc.returncode != 0:
            print(f"[WARN] llm_demo failed for {prompt_path} (code {proc.returncode})", file=sys.stderr)

    print(f"Runs logged to {args.log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
