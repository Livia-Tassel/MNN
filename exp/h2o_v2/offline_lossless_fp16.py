#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np

try:
    import zstandard as zstd
except Exception:
    zstd = None


def iter_meta_files(root):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.startswith("meta_") and name.endswith(".json"):
                yield Path(dirpath) / name


def parse_layer_ids(text):
    ids = set()
    if not text:
        return ids
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        ids.add(int(part))
    return ids


def should_keep_stage(meta_stage, target_stage):
    if target_stage == "both":
        return True
    return meta_stage == target_stage


def should_keep_layer(layer_id, args, layer_id_set):
    if args.scope == "front_n":
        return 0 <= layer_id < args.front_n
    if args.scope == "layer_range":
        if layer_id < args.layer_start:
            return False
        if args.layer_end >= 0 and layer_id > args.layer_end:
            return False
        return True
    if args.scope == "layer_ids":
        return layer_id in layer_id_set
    return False


def delta_stream_u8(arr_u8, seq_len, heads, head_dim):
    if seq_len <= 1:
        return arr_u8.tobytes()
    view = arr_u8.reshape((seq_len, heads, head_dim))
    first = view[0:1]
    delta = (view[1:].astype(np.int16) - view[:-1].astype(np.int16)) & 0xFF
    delta = delta.astype(np.uint8)
    return np.concatenate([first, delta], axis=0).tobytes()


def gear_delta_compressed_size_fp16(fp16_bytes, seq_len, heads, head_dim, compressor):
    expected_bytes = seq_len * heads * head_dim * 2
    if len(fp16_bytes) != expected_bytes:
        raise ValueError(
            f"size mismatch: got={len(fp16_bytes)} expected={expected_bytes} "
            f"(seq={seq_len}, heads={heads}, dim={head_dim})"
        )
    u16 = np.frombuffer(fp16_bytes, dtype=np.uint16)
    hi = (u16 >> 8).astype(np.uint8)
    lo = (u16 & 0xFF).astype(np.uint8)
    hi_delta_bytes = delta_stream_u8(hi, seq_len, heads, head_dim)
    lo_bytes = lo.tobytes()
    hi_delta_c = compressor.compress(hi_delta_bytes)
    lo_c = compressor.compress(lo_bytes)
    return len(hi_delta_c) + len(lo_c)


def normalize_to_fp16(data, bytes_per_elem, strict_fp16):
    if bytes_per_elem == 2:
        return data, "fp16"
    if bytes_per_elem == 4:
        if strict_fp16:
            return None, "skip_fp32_strict"
        f32 = np.frombuffer(data, dtype=np.float32)
        f16 = f32.astype(np.float16)
        return f16.tobytes(), "fp32_to_fp16"
    return None, f"unsupported_bytes_per_elem_{bytes_per_elem}"


def ratio(raw_bytes, compressed_bytes):
    if compressed_bytes <= 0:
        return 0.0
    return float(raw_bytes) / float(compressed_bytes)


def write_markdown(path, result):
    with Path(path).open("w", encoding="utf-8") as f:
        f.write("# H2O v2 Offline Lossless Summary\n\n")
        f.write(f"- Dump dir: `{result['dump_dir']}`\n")
        f.write(f"- Stage: `{result['stage']}`\n")
        f.write(f"- Scope: `{result['scope']}`\n")
        f.write(f"- Selected entries: {result['selected_entries']}\n")
        f.write(f"- Selected layers: {result['selected_layers']}\n")
        f.write(f"- Selected dumps: {result['selected_dumps']}\n")
        f.write(f"- Zstd level: {result['zstd_level']}\n")
        f.write(f"- Codec: `{result['codec']}`\n\n")
        f.write("## Ratios\n\n")
        f.write(f"- K lossless ratio: {result['k_lossless_ratio']:.4f}\n")
        f.write(f"- V lossless ratio: {result['v_lossless_ratio']:.4f}\n")
        f.write(f"- Overall lossless ratio: {result['lossless_ratio']:.4f}\n")
        f.write(f"- Weighted lossless ratio: {result['weighted_lossless_ratio']:.4f}\n\n")
        f.write("## Bytes\n\n")
        f.write(f"- K raw bytes: {result['k_raw_bytes']}\n")
        f.write(f"- K compressed bytes: {result['k_compressed_bytes']}\n")
        f.write(f"- V raw bytes: {result['v_raw_bytes']}\n")
        f.write(f"- V compressed bytes: {result['v_compressed_bytes']}\n")
        f.write(f"- Total raw bytes: {result['raw_bytes']}\n")
        f.write(f"- Total compressed bytes: {result['compressed_bytes']}\n\n")
        f.write("## Diagnostics\n\n")
        f.write(f"- Scanned entries: {result['scanned_entries']}\n")
        f.write(f"- Converted fp32->fp16 entries: {result['converted_fp32_entries']}\n")
        f.write(f"- Skipped entries: {result['skipped_entries']}\n")
        if result["skip_reasons"]:
            f.write("- Skip reasons:\n")
            for reason, count in sorted(result["skip_reasons"].items()):
                f.write(f"  - {reason}: {count}\n")


def main():
    parser = argparse.ArgumentParser(description="Offline lossless ratio eval for H2O v2 front layers (FP16 gear+delta+zstd).")
    parser.add_argument("--dump-dir", required=True, help="Root directory containing meta_*.json and KV binary files.")
    parser.add_argument("--out-json", required=True, help="Output json path for offline lossless metrics.")
    parser.add_argument("--out-md", default="", help="Optional markdown summary path.")
    parser.add_argument("--stage", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--scope", choices=["front_n", "layer_range", "layer_ids"], default="front_n")
    parser.add_argument("--front-n", type=int, default=2, help="Used when scope=front_n, keep layers [0, front_n).")
    parser.add_argument("--layer-start", type=int, default=2, help="Used when scope=layer_range.")
    parser.add_argument("--layer-end", type=int, default=-1, help="Used when scope=layer_range, -1 means no upper bound.")
    parser.add_argument("--layer-ids", default="", help="Comma separated list, used when scope=layer_ids.")
    parser.add_argument("--zstd-level", type=int, default=3)
    parser.add_argument("--strict-fp16", action="store_true", help="If set, skip entries with bytes_per_elem=4 instead of converting.")
    parser.add_argument("--min-seq-len", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if zstd is None:
        raise SystemExit("Missing dependency: zstandard. Install with `pip install zstandard`.")
    if args.scope == "front_n" and args.front_n <= 0:
        raise SystemExit("--front-n must be > 0 when scope=front_n.")

    layer_id_set = parse_layer_ids(args.layer_ids)
    if args.scope == "layer_ids" and not layer_id_set:
        raise SystemExit("--layer-ids must be non-empty when scope=layer_ids.")

    meta_files = sorted(iter_meta_files(args.dump_dir))
    if not meta_files:
        raise SystemExit(f"No meta_*.json found under {args.dump_dir}")

    compressor = zstd.ZstdCompressor(level=args.zstd_level)

    scanned_entries = 0
    selected_entries = 0
    converted_fp32_entries = 0
    selected_layers = set()
    selected_dumps = set()
    skipped_entries = 0
    skip_reasons = {}

    k_raw_bytes = 0
    k_compressed_bytes = 0
    v_raw_bytes = 0
    v_compressed_bytes = 0

    def skip(reason):
        nonlocal skipped_entries
        skipped_entries += 1
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    for meta_path in meta_files:
        scanned_entries += 1
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            skip("meta_parse_error")
            continue

        stage = str(meta.get("stage", "unknown"))
        if not should_keep_stage(stage, args.stage):
            continue

        try:
            layer_id = int(meta.get("layer_id", -1))
            seq_len = int(meta.get("seq_len", 0))
            kv_heads = int(meta.get("kv_heads", 0))
            head_dim = int(meta.get("head_dim", 0))
            bytes_per_elem = int(meta.get("bytes_per_elem", 0))
        except Exception:
            skip("meta_invalid_numeric_fields")
            continue

        if not should_keep_layer(layer_id, args, layer_id_set):
            continue
        if seq_len <= 0 or kv_heads <= 0 or head_dim <= 0:
            skip("invalid_shape")
            continue
        if args.min_seq_len and seq_len < args.min_seq_len:
            continue
        if args.max_seq_len and seq_len > args.max_seq_len:
            continue

        k_file = meta.get("k_file", "")
        v_file = meta.get("v_file", "")
        if not isinstance(k_file, str) or not k_file or not isinstance(v_file, str) or not v_file:
            skip("missing_k_or_v_file")
            continue
        k_path = (meta_path.parent / k_file).resolve()
        v_path = (meta_path.parent / v_file).resolve()
        if not k_path.exists() or not v_path.exists():
            skip("missing_k_or_v_binary")
            continue

        try:
            k_bytes = k_path.read_bytes()
            v_bytes = v_path.read_bytes()
        except Exception:
            skip("binary_read_error")
            continue

        k_fp16, k_norm_status = normalize_to_fp16(k_bytes, bytes_per_elem, args.strict_fp16)
        v_fp16, v_norm_status = normalize_to_fp16(v_bytes, bytes_per_elem, args.strict_fp16)
        if k_fp16 is None or v_fp16 is None:
            skip(k_norm_status if k_fp16 is None else v_norm_status)
            continue
        if k_norm_status == "fp32_to_fp16" or v_norm_status == "fp32_to_fp16":
            converted_fp32_entries += 1

        try:
            k_comp = gear_delta_compressed_size_fp16(k_fp16, seq_len, kv_heads, head_dim, compressor)
            v_comp = gear_delta_compressed_size_fp16(v_fp16, seq_len, kv_heads, head_dim, compressor)
        except Exception:
            skip("compression_failed")
            continue

        selected_entries += 1
        selected_layers.add(layer_id)
        run_id = meta.get("run_id")
        if isinstance(run_id, str) and run_id:
            selected_dumps.add(run_id)
        else:
            selected_dumps.add(str(meta_path.parent))

        k_raw_bytes += len(k_fp16)
        k_compressed_bytes += k_comp
        v_raw_bytes += len(v_fp16)
        v_compressed_bytes += v_comp

    if selected_entries == 0:
        raise SystemExit("No entries selected after filtering. Check --dump-dir/--stage/--scope options.")

    raw_bytes = k_raw_bytes + v_raw_bytes
    compressed_bytes = k_compressed_bytes + v_compressed_bytes
    k_lossless_ratio = ratio(k_raw_bytes, k_compressed_bytes)
    v_lossless_ratio = ratio(v_raw_bytes, v_compressed_bytes)
    lossless_ratio = ratio(raw_bytes, compressed_bytes)

    result = {
        "schema_version": 1,
        "dump_dir": str(Path(args.dump_dir).resolve()),
        "stage": args.stage,
        "scope": args.scope,
        "front_n": int(args.front_n),
        "layer_start": int(args.layer_start),
        "layer_end": int(args.layer_end),
        "layer_ids": sorted(layer_id_set),
        "selected_layers": sorted(selected_layers),
        "selected_entries": int(selected_entries),
        "selected_dumps": int(len(selected_dumps)),
        "codec": "fp16_gear_delta_zstd",
        "zstd_level": int(args.zstd_level),
        "k_raw_bytes": int(k_raw_bytes),
        "k_compressed_bytes": int(k_compressed_bytes),
        "v_raw_bytes": int(v_raw_bytes),
        "v_compressed_bytes": int(v_compressed_bytes),
        "raw_bytes": int(raw_bytes),
        "compressed_bytes": int(compressed_bytes),
        "k_lossless_ratio": float(k_lossless_ratio),
        "v_lossless_ratio": float(v_lossless_ratio),
        "lossless_ratio": float(lossless_ratio),
        "weighted_lossless_ratio": float(lossless_ratio),
        "scanned_entries": int(scanned_entries),
        "converted_fp32_entries": int(converted_fp32_entries),
        "skipped_entries": int(skipped_entries),
        "skip_reasons": skip_reasons,
    }

    out_json_path = Path(args.out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote {out_json_path}")

    if args.out_md:
        out_md_path = Path(args.out_md)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(out_md_path, result)
        print(f"Wrote {out_md_path}")

    if args.verbose:
        print(
            f"[offline_lossless_fp16] selected={selected_entries} "
            f"layers={len(selected_layers)} ratio={lossless_ratio:.4f} "
            f"k={k_lossless_ratio:.4f} v={v_lossless_ratio:.4f}"
        )


if __name__ == "__main__":
    main()
