#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None

try:
    import zstandard as zstd
except Exception:
    zstd = None


VALID_PREDICTORS = {
    "raw",
    "delta_seq",
    "xor_seq",
    "pair_delta",
    "pair_delta_delta_seq",
}


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


def parse_predictor_list(text, arg_name):
    items = []
    seen = set()
    for part in text.split(","):
        name = part.strip()
        if not name:
            continue
        if name not in VALID_PREDICTORS:
            raise SystemExit(f"{arg_name}: invalid predictor `{name}`")
        if name not in seen:
            items.append(name)
            seen.add(name)
    if not items:
        raise SystemExit(f"{arg_name}: predictor list is empty")
    return items


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


def delta_seq_block_u8(block):
    seq_len = block.shape[0]
    if seq_len <= 1:
        return block
    first = block[0:1]
    delta = (block[1:].astype(np.int16) - block[:-1].astype(np.int16)) & 0xFF
    delta = delta.astype(np.uint8)
    return np.concatenate([first, delta], axis=0)


def xor_seq_block_u8(block):
    seq_len = block.shape[0]
    if seq_len <= 1:
        return block
    first = block[0:1]
    xored = np.bitwise_xor(block[1:], block[:-1])
    return np.concatenate([first, xored], axis=0)


def pair_delta_block_u8(block):
    head_dim = block.shape[-1]
    if head_dim <= 1:
        return block
    out = block.copy()
    left = out[..., 0::2]
    right = out[..., 1::2]
    if right.size == 0:
        return out
    left_base = left[..., : right.shape[-1]]
    right_delta = (right.astype(np.int16) - left_base.astype(np.int16)) & 0xFF
    out[..., 1::2] = right_delta.astype(np.uint8)
    return out


def apply_predictor(block_u8, predictor):
    if predictor == "raw":
        return block_u8
    if predictor == "delta_seq":
        return delta_seq_block_u8(block_u8)
    if predictor == "xor_seq":
        return xor_seq_block_u8(block_u8)
    if predictor == "pair_delta":
        return pair_delta_block_u8(block_u8)
    if predictor == "pair_delta_delta_seq":
        return delta_seq_block_u8(pair_delta_block_u8(block_u8))
    raise ValueError(f"unsupported predictor: {predictor}")


def bitshuffle_u8_bytes(data_bytes):
    if not data_bytes:
        return data_bytes
    arr = np.frombuffer(data_bytes, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="little").reshape(arr.size, 8)
    shuffled = bits.T.reshape(-1)
    return np.packbits(shuffled, bitorder="little").tobytes()


def compress_stream_legacy(stream_u8, predictor, compressor, use_bitshuffle):
    transformed = apply_predictor(stream_u8, predictor)
    payload = np.ascontiguousarray(transformed).reshape(-1).tobytes()
    if use_bitshuffle:
        payload = bitshuffle_u8_bytes(payload)
    compressed_size = len(compressor.compress(payload))
    diag = {
        "mode": "legacy",
        "predictor": predictor,
        "blocks": 1,
        "mode_counts": {predictor: 1},
        "data_compressed_bytes": compressed_size,
        "mode_bytes": 0,
        "header_bytes": 0,
        "stream_total_bytes": compressed_size,
    }
    return compressed_size, diag


def compress_stream_adaptive(stream_u8, predictors, block_seq, compressor, use_bitshuffle):
    seq_len = int(stream_u8.shape[0])
    if seq_len <= 0:
        return 0, {
            "mode": "adaptive_v22",
            "predictors": predictors,
            "blocks": 0,
            "mode_counts": {name: 0 for name in predictors},
            "data_compressed_bytes": 0,
            "mode_bytes": 0,
            "header_bytes": 0,
            "stream_total_bytes": 0,
        }

    real_block_seq = max(1, min(int(block_seq), seq_len))
    transformed_chunks = []
    mode_ids = []
    mode_counts = {name: 0 for name in predictors}

    for start in range(0, seq_len, real_block_seq):
        block = stream_u8[start : start + real_block_seq]
        best_name = None
        best_payload = None
        best_size = None
        for name in predictors:
            transformed = apply_predictor(block, name)
            payload = np.ascontiguousarray(transformed).reshape(-1).tobytes()
            if use_bitshuffle:
                payload = bitshuffle_u8_bytes(payload)
            size = len(compressor.compress(payload))
            if best_size is None or size < best_size:
                best_size = size
                best_name = name
                best_payload = payload
        transformed_chunks.append(best_payload)
        mode_ids.append(predictors.index(best_name))
        mode_counts[best_name] += 1

    merged_payload = b"".join(transformed_chunks)
    data_compressed_bytes = len(compressor.compress(merged_payload)) if merged_payload else 0
    mode_bytes = len(mode_ids)  # one mode byte per block
    header_bytes = 8            # block_seq + predictor_count + reserved metadata
    stream_total_bytes = data_compressed_bytes + mode_bytes + header_bytes

    diag = {
        "mode": "adaptive_v22",
        "predictors": predictors,
        "block_seq": real_block_seq,
        "blocks": len(mode_ids),
        "mode_counts": mode_counts,
        "data_compressed_bytes": data_compressed_bytes,
        "mode_bytes": mode_bytes,
        "header_bytes": header_bytes,
        "stream_total_bytes": stream_total_bytes,
    }
    return stream_total_bytes, diag


def compress_tensor_fp16(fp16_bytes, seq_len, heads, head_dim, compressor, args, tensor_name):
    expected_bytes = seq_len * heads * head_dim * 2
    if len(fp16_bytes) != expected_bytes:
        raise ValueError(
            f"{tensor_name}: size mismatch got={len(fp16_bytes)} expected={expected_bytes} "
            f"(seq={seq_len}, heads={heads}, dim={head_dim})"
        )

    u16 = np.frombuffer(fp16_bytes, dtype=np.uint16).reshape((seq_len, heads, head_dim))
    hi = ((u16 >> 8) & 0xFF).astype(np.uint8)
    lo = (u16 & 0xFF).astype(np.uint8)

    if args.codec_mode == "legacy":
        hi_size, hi_diag = compress_stream_legacy(
            hi, predictor="delta_seq", compressor=compressor, use_bitshuffle=args.bitshuffle
        )
        lo_size, lo_diag = compress_stream_legacy(
            lo, predictor="raw", compressor=compressor, use_bitshuffle=args.bitshuffle
        )
    else:
        if tensor_name == "k":
            hi_predictors = args.k_hi_predictors_list
            lo_predictors = args.k_lo_predictors_list
        else:
            hi_predictors = args.v_hi_predictors_list
            lo_predictors = args.v_lo_predictors_list
        hi_size, hi_diag = compress_stream_adaptive(
            hi,
            predictors=hi_predictors,
            block_seq=args.adaptive_block_seq,
            compressor=compressor,
            use_bitshuffle=args.bitshuffle,
        )
        lo_size, lo_diag = compress_stream_adaptive(
            lo,
            predictors=lo_predictors,
            block_seq=args.adaptive_block_seq,
            compressor=compressor,
            use_bitshuffle=args.bitshuffle,
        )

    return hi_size + lo_size, {"hi": hi_diag, "lo": lo_diag}


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


def ensure_not_overwrite(path, overwrite, label):
    p = Path(path)
    if p.exists() and not overwrite:
        raise SystemExit(
            f"{label} already exists: {p}\n"
            f"Use a new output path or pass --overwrite."
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def merge_mode_counts(dst, src):
    for mode_name, count in src.items():
        dst[mode_name] = dst.get(mode_name, 0) + int(count)


def write_markdown(path, result):
    with Path(path).open("w", encoding="utf-8") as f:
        f.write("# H2O v2 Offline Lossless Summary\n\n")
        f.write(f"- Dump dir: `{result['dump_dir']}`\n")
        f.write(f"- Stage: `{result['stage']}`\n")
        f.write(f"- Scope: `{result['scope']}`\n")
        f.write(f"- Selected entries: {result['selected_entries']}\n")
        f.write(f"- Selected layers: {result['selected_layers']}\n")
        f.write(f"- Selected dumps: {result['selected_dumps']}\n")
        f.write(f"- Codec mode: `{result['codec_mode']}`\n")
        f.write(f"- Bitshuffle: `{result['bitshuffle']}`\n")
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
        f.write("## Predictor Usage (Adaptive)\n\n")
        f.write(f"- K hi mode counts: {result['k_hi_mode_counts']}\n")
        f.write(f"- K lo mode counts: {result['k_lo_mode_counts']}\n")
        f.write(f"- V hi mode counts: {result['v_hi_mode_counts']}\n")
        f.write(f"- V lo mode counts: {result['v_lo_mode_counts']}\n\n")
        f.write("## Diagnostics\n\n")
        f.write(f"- Scanned entries: {result['scanned_entries']}\n")
        f.write(f"- Converted fp32->fp16 entries: {result['converted_fp32_entries']}\n")
        f.write(f"- Skipped entries: {result['skipped_entries']}\n")
        if result["skip_reasons"]:
            f.write("- Skip reasons:\n")
            for reason, count in sorted(result["skip_reasons"].items()):
                f.write(f"  - {reason}: {count}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Offline lossless ratio eval for H2O v2 front layers (FP16, v2.2 adaptive lossless codec)."
    )
    parser.add_argument("--dump-dir", required=True, help="Root directory containing meta_*.json and KV binary files.")
    parser.add_argument("--out-json", required=True, help="Output json path for offline lossless metrics.")
    parser.add_argument("--out-md", default="", help="Optional markdown summary path.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing out-json/out-md.")
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
    parser.add_argument("--codec-mode", choices=["legacy", "adaptive_v22"], default="adaptive_v22")
    parser.add_argument("--adaptive-block-seq", type=int, default=64, help="Seq block size for adaptive predictor selection.")
    parser.add_argument("--bitshuffle", action="store_true", help="Apply bitshuffle before zstd on transformed streams.")
    parser.add_argument(
        "--k-hi-predictors",
        default="raw,delta_seq,xor_seq,pair_delta,pair_delta_delta_seq",
        help="Comma list for K-hi predictor candidates (adaptive mode).",
    )
    parser.add_argument(
        "--k-lo-predictors",
        default="raw,delta_seq,xor_seq,pair_delta,pair_delta_delta_seq",
        help="Comma list for K-lo predictor candidates (adaptive mode).",
    )
    parser.add_argument(
        "--v-hi-predictors",
        default="raw,delta_seq,xor_seq",
        help="Comma list for V-hi predictor candidates (adaptive mode).",
    )
    parser.add_argument(
        "--v-lo-predictors",
        default="raw,delta_seq,xor_seq",
        help="Comma list for V-lo predictor candidates (adaptive mode).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if zstd is None:
        raise SystemExit("Missing dependency: zstandard. Install with `pip install zstandard`.")
    if np is None:
        raise SystemExit("Missing dependency: numpy. Install with `pip install numpy`.")
    if args.scope == "front_n" and args.front_n <= 0:
        raise SystemExit("--front-n must be > 0 when scope=front_n.")
    if args.adaptive_block_seq <= 0:
        raise SystemExit("--adaptive-block-seq must be > 0.")

    args.k_hi_predictors_list = parse_predictor_list(args.k_hi_predictors, "--k-hi-predictors")
    args.k_lo_predictors_list = parse_predictor_list(args.k_lo_predictors, "--k-lo-predictors")
    args.v_hi_predictors_list = parse_predictor_list(args.v_hi_predictors, "--v-hi-predictors")
    args.v_lo_predictors_list = parse_predictor_list(args.v_lo_predictors, "--v-lo-predictors")

    layer_id_set = parse_layer_ids(args.layer_ids)
    if args.scope == "layer_ids" and not layer_id_set:
        raise SystemExit("--layer-ids must be non-empty when scope=layer_ids.")

    out_json_path = ensure_not_overwrite(args.out_json, args.overwrite, "out-json")
    out_md_path = None
    if args.out_md:
        out_md_path = ensure_not_overwrite(args.out_md, args.overwrite, "out-md")

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

    k_hi_mode_counts = {}
    k_lo_mode_counts = {}
    v_hi_mode_counts = {}
    v_lo_mode_counts = {}

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
            k_comp, k_diag = compress_tensor_fp16(k_fp16, seq_len, kv_heads, head_dim, compressor, args, tensor_name="k")
            v_comp, v_diag = compress_tensor_fp16(v_fp16, seq_len, kv_heads, head_dim, compressor, args, tensor_name="v")
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

        merge_mode_counts(k_hi_mode_counts, k_diag["hi"]["mode_counts"])
        merge_mode_counts(k_lo_mode_counts, k_diag["lo"]["mode_counts"])
        merge_mode_counts(v_hi_mode_counts, v_diag["hi"]["mode_counts"])
        merge_mode_counts(v_lo_mode_counts, v_diag["lo"]["mode_counts"])

    if selected_entries == 0:
        raise SystemExit("No entries selected after filtering. Check --dump-dir/--stage/--scope options.")

    raw_bytes = k_raw_bytes + v_raw_bytes
    compressed_bytes = k_compressed_bytes + v_compressed_bytes
    k_lossless_ratio = ratio(k_raw_bytes, k_compressed_bytes)
    v_lossless_ratio = ratio(v_raw_bytes, v_compressed_bytes)
    lossless_ratio = ratio(raw_bytes, compressed_bytes)

    result = {
        "schema_version": 2,
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
        "codec": "fp16_gear_predictive_zstd",
        "codec_mode": args.codec_mode,
        "bitshuffle": bool(args.bitshuffle),
        "adaptive_block_seq": int(args.adaptive_block_seq),
        "k_hi_predictors": args.k_hi_predictors_list,
        "k_lo_predictors": args.k_lo_predictors_list,
        "v_hi_predictors": args.v_hi_predictors_list,
        "v_lo_predictors": args.v_lo_predictors_list,
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
        "k_hi_mode_counts": k_hi_mode_counts,
        "k_lo_mode_counts": k_lo_mode_counts,
        "v_hi_mode_counts": v_hi_mode_counts,
        "v_lo_mode_counts": v_lo_mode_counts,
        "scanned_entries": int(scanned_entries),
        "converted_fp32_entries": int(converted_fp32_entries),
        "skipped_entries": int(skipped_entries),
        "skip_reasons": skip_reasons,
    }

    out_json_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote {out_json_path}")

    if out_md_path is not None:
        write_markdown(out_md_path, result)
        print(f"Wrote {out_md_path}")

    if args.verbose:
        print(
            f"[offline_lossless_fp16] selected={selected_entries} "
            f"layers={len(selected_layers)} ratio={lossless_ratio:.4f} "
            f"k={k_lossless_ratio:.4f} v={v_lossless_ratio:.4f}"
        )
        print(
            f"[offline_lossless_fp16] mode={args.codec_mode} "
            f"bitshuffle={int(args.bitshuffle)} block_seq={args.adaptive_block_seq}"
        )


if __name__ == "__main__":
    main()
