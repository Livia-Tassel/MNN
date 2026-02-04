#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import sys
from typing import Callable, Dict, Iterable, List, Optional, Tuple


def format_mmlu(obj: Dict) -> Optional[str]:
    question = obj.get("question") or obj.get("prompt") or obj.get("input")
    choices = obj.get("choices") or obj.get("options")
    if not question or not choices:
        return None
    lines = [f"Q: {str(question).strip()}"]
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, choice in enumerate(choices):
        label = labels[i] if i < len(labels) else f"Choice{i+1}"
        lines.append(f"{label}. {str(choice).strip()}")
    return "\n".join(lines)


def format_generic(obj: Dict) -> Optional[str]:
    for key in ("text", "prompt", "input", "question"):
        if key in obj and obj[key]:
            return str(obj[key]).strip()
    return None


def load_jsonl(path: str, jsonl_format: str) -> List[str]:
    items = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = None
            if jsonl_format == "mmlu":
                text = format_mmlu(obj)
            elif jsonl_format == "text":
                text = format_generic(obj)
            else:
                text = format_mmlu(obj) or format_generic(obj)
            if text:
                items.append(text)
    return items


def split_pieces(text: str, min_chars: int) -> List[str]:
    parts = re.split(r"\n\\s*\n", text.replace("\r", ""))
    pieces = []
    for part in parts:
        p = part.strip()
        if len(p) >= min_chars:
            pieces.append(p)
    return pieces


def load_corpus(paths: List[str], min_chars: int, jsonl_format: str) -> List[str]:
    pieces: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                full = os.path.join(path, name)
                if name.endswith(".jsonl") or name.endswith(".json"):
                    pieces.extend(load_jsonl(full, jsonl_format))
                elif name.endswith(".txt"):
                    with open(full, "r", encoding="utf-8", errors="ignore") as f:
                        pieces.extend(split_pieces(f.read(), min_chars))
        else:
            if path.endswith(".jsonl") or path.endswith(".json"):
                pieces.extend(load_jsonl(path, jsonl_format))
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    pieces.extend(split_pieces(f.read(), min_chars))
    return pieces


def build_token_counter(mode: str, config: str, hf_path: Optional[str], strict: bool) -> Tuple[Callable[[str], int], str]:
    if mode in ("auto", "mnn"):
        try:
            import MNN.llm as mnnllm  # type: ignore
            llm = mnnllm.create(config)
            llm.load()
            def count_mnn(text: str) -> int:
                return len(llm.tokenizer_encode(text))
            return count_mnn, "mnn"
        except Exception as exc:
            if mode == "mnn":
                raise RuntimeError(f"Failed to initialize MNN tokenizer: {exc}") from exc

    if mode in ("auto", "hf"):
        try:
            from transformers import AutoTokenizer  # type: ignore
            model_dir = hf_path or os.path.dirname(os.path.abspath(config))
            tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True, local_files_only=True)
            def count_hf(text: str) -> int:
                return len(tokenizer.encode(text, add_special_tokens=False))
            return count_hf, "hf"
        except Exception as exc:
            if mode == "hf":
                raise RuntimeError(f"Failed to initialize HF tokenizer: {exc}") from exc

    if strict:
        raise RuntimeError("Tokenizer fallback not allowed in strict mode.")

    def count_approx(text: str) -> int:
        return len(text.split())

    return count_approx, "approx"


def trim_to_target(text: str, count_fn: Callable[[str], int], target: int, tolerance: float) -> str:
    words = text.split()
    if not words:
        return text
    low, high = 1, len(words)
    best = None
    while low <= high:
        mid = (low + high) // 2
        candidate = " ".join(words[:mid])
        tokens = count_fn(candidate)
        if tokens >= target * (1 - tolerance) and tokens <= target * (1 + tolerance):
            best = candidate
            break
        if tokens < target:
            low = mid + 1
            best = candidate
        else:
            high = mid - 1
            best = candidate
    return best or text


def build_prompt(pieces: List[str],
                 cursor: int,
                 target_tokens: int,
                 tolerance: float,
                 count_fn: Callable[[str], int]) -> Tuple[str, int, int]:
    parts = []
    total_tokens = 0
    start_cursor = cursor
    while True:
        piece = pieces[cursor % len(pieces)]
        candidate = "\\n\\n".join(parts + [piece]) if parts else piece
        total_tokens = count_fn(candidate)
        parts.append(piece)
        cursor += 1
        if total_tokens >= target_tokens:
            break
        if cursor - start_cursor >= len(pieces):
            break

    if total_tokens > target_tokens * (1 + tolerance):
        last = parts[-1]
        prefix = "\\n\\n".join(parts[:-1]) if len(parts) > 1 else ""
        remaining = target_tokens - (count_fn(prefix) if prefix else 0)
        trimmed = trim_to_target(last, count_fn, max(1, remaining), tolerance)
        if prefix:
            candidate = prefix + "\\n\\n" + trimmed
        else:
            candidate = trimmed
        total_tokens = count_fn(candidate)
        return candidate, total_tokens, cursor

    return "\\n\\n".join(parts), total_tokens, cursor


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate prompt files with target token lengths.")
    parser.add_argument("--config", required=True, help="Model config.json path (for tokenizer).")
    parser.add_argument("--corpus", nargs="+", required=True, help="Text/JSONL file(s) or directories.")
    parser.add_argument("--out-dir", default="exp/gear_fp16/prompts", help="Output directory for prompt files.")
    parser.add_argument("--targets", default="128,512,2048", help="Comma-separated target token lengths.")
    parser.add_argument("--per-target", type=int, default=3, help="Prompts to generate per target.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-piece-chars", type=int, default=200, help="Only applies to plain text files.")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Token count tolerance.")
    parser.add_argument("--tokenizer", choices=["auto", "mnn", "hf", "approx"], default="auto")
    parser.add_argument("--jsonl-format", choices=["auto", "mmlu", "text"], default="auto",
                        help="How to parse JSONL rows.")
    parser.add_argument("--hf-path", default=None, help="Optional HF tokenizer directory.")
    parser.add_argument("--strict", action="store_true", help="Fail if tokenizer fallback is needed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prompt files.")
    args = parser.parse_args()

    targets = [int(t.strip()) for t in args.targets.split(",") if t.strip()]
    if not targets:
        print("No valid targets provided.", file=sys.stderr)
        return 1

    count_fn, tokenizer_used = build_token_counter(args.tokenizer, args.config, args.hf_path, args.strict)

    pieces = load_corpus(args.corpus, args.min_piece_chars, args.jsonl_format)
    if not pieces:
        print("No pieces found. Reduce --min-piece-chars.", file=sys.stderr)
        return 1

    random.seed(args.seed)
    random.shuffle(pieces)

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "prompts_manifest.jsonl")

    cursor = 0
    written = 0
    for target in targets:
        for idx in range(args.per_target):
            prompt, tokens, cursor = build_prompt(pieces, cursor, target, args.tolerance, count_fn)
            filename = f"prompt_{target}_{idx+1}.txt"
            path = os.path.join(args.out_dir, filename)
            if os.path.exists(path) and not args.overwrite:
                print(f"Skip existing {path} (use --overwrite).")
                continue
            with open(path, "w", encoding="utf-8") as f:
                f.write(prompt)

            record = {
                "file": path,
                "target_tokens": target,
                "actual_tokens": tokens,
                "tokenizer": tokenizer_used,
                "chars": len(prompt),
            }
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\\n")
            written += 1

    print(f"Tokenizer: {tokenizer_used}")
    print(f"Wrote {written} prompts to {args.out_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
