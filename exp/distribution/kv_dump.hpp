// Experimental KV dump utilities for distribution analysis.
// This header is intended to be included in a single translation unit.

#ifndef MNN_EXP_DISTRIBUTION_KV_DUMP_HPP
#define MNN_EXP_DISTRIBUTION_KV_DUMP_HPP

#include <MNN/Tensor.hpp>
#include "core/MNNFileUtils.h"
#include "core/OpCommonUtils.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace MNN {
namespace KVExp {

enum class DumpStage {
    Prefill,
    Decode,
    Both,
};

struct DumpConfig {
    bool enabled = false;
    bool dump_packed = false;
    std::string root_dir;
    std::string run_id;
    int max_tokens = 0;
    DumpStage stage = DumpStage::Prefill;
};

inline std::string GetEnvString(const char* name, const std::string& default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return std::string(value);
}

inline int GetEnvInt(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return std::atoi(value);
}

inline DumpStage ParseStage(const std::string& stage) {
    if (stage == "decode") {
        return DumpStage::Decode;
    }
    if (stage == "both") {
        return DumpStage::Both;
    }
    return DumpStage::Prefill;
}

inline bool ParseBool(const std::string& value) {
    if (value == "1" || value == "true" || value == "on" || value == "yes") {
        return true;
    }
    return false;
}

inline std::string MakeRunId() {
    static std::atomic<uint64_t> counter{0};
    std::ostringstream oss;
    oss << "run_" << static_cast<uint64_t>(std::time(nullptr)) << "_" << counter.fetch_add(1);
    return oss.str();
}

inline DumpConfig& GetConfig() {
    static DumpConfig config = []() {
        DumpConfig cfg;
        std::string root = GetEnvString("MNN_KV_DUMP_DIR", "exp/distribution/dumps");
        if (root == "0" || root == "false" || root == "off") {
            cfg.enabled = false;
            return cfg;
        }
        cfg.enabled = true;
        cfg.root_dir = root;
        cfg.run_id = GetEnvString("MNN_KV_DUMP_RUN_ID", "");
        if (cfg.run_id.empty()) {
            cfg.run_id = MakeRunId();
        }
        cfg.max_tokens = GetEnvInt("MNN_KV_DUMP_MAX_TOKENS", 0);
        cfg.stage = ParseStage(GetEnvString("MNN_KV_DUMP_STAGE", "prefill"));
        cfg.dump_packed = ParseBool(GetEnvString("MNN_KV_DUMP_PACKED", "0"));
        return cfg;
    }();
    return config;
}

inline bool EnsureDirRecursive(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    if (MNNDirExist(path.c_str())) {
        return true;
    }
    std::string current;
    current.reserve(path.size());
    size_t i = 0;
    if (path.size() >= 1 && (path[0] == '/' || path[0] == '\\')) {
        current.push_back(path[0]);
        i = 1;
    }
    for (; i < path.size(); ++i) {
        char c = path[i];
        current.push_back(c);
        if (c == '/' || c == '\\') {
            if (current.size() == 1) {
                continue;
            }
            if (!MNNCreateDir(current.c_str())) {
                return false;
            }
        }
    }
    if (!MNNCreateDir(current.c_str())) {
        return MNNDirExist(current.c_str());
    }
    return true;
}

inline int GetLayerId(const void* layer_ptr, int /* layer_nums */) {
    static std::mutex guard;
    static std::unordered_map<const void*, int> map;
    static int next_id = 0;
    std::lock_guard<std::mutex> lock(guard);
    auto it = map.find(layer_ptr);
    if (it != map.end()) {
        return it->second;
    }
    int id = next_id++;
    map.emplace(layer_ptr, id);
    return id;
}

inline bool ShouldDump(DumpStage stage, int add) {
    if (stage == DumpStage::Both) {
        return true;
    }
    if (stage == DumpStage::Decode) {
        return add == 1;
    }
    return add > 1;
}

inline void WriteBinaryFile(const std::string& path, const uint8_t* data, size_t size) {
    std::ofstream out(path.c_str(), std::ios::binary);
    if (!out) {
        return;
    }
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
}

inline void WriteMetaFile(const std::string& path,
                          int layer_id,
                          int seq_start,
                          int seq_len,
                          int kv_heads,
                          int head_dim,
                          int bytes_per_elem,
                          int add,
                          int tensor_seq_len,
                          bool quant_key,
                          bool quant_value,
                          bool flash_attention,
                          const std::string& stage,
                          const std::string& run_id,
                          const std::string& k_file,
                          const std::string& v_file) {
    std::ofstream out(path.c_str(), std::ios::out);
    if (!out) {
        return;
    }
    out << "{\n";
    out << "  \"layer_id\": " << layer_id << ",\n";
    out << "  \"seq_start\": " << seq_start << ",\n";
    out << "  \"seq_len\": " << seq_len << ",\n";
    out << "  \"seq_end_exclusive\": " << (seq_start + seq_len) << ",\n";
    out << "  \"kv_heads\": " << kv_heads << ",\n";
    out << "  \"head_dim\": " << head_dim << ",\n";
    out << "  \"bytes_per_elem\": " << bytes_per_elem << ",\n";
    out << "  \"dtype\": \"" << (bytes_per_elem == 2 ? "fp16" : (bytes_per_elem == 4 ? "fp32" : "unknown")) << "\",\n";
    out << "  \"add\": " << add << ",\n";
    out << "  \"tensor_seq_len\": " << tensor_seq_len << ",\n";
    out << "  \"quant_key\": " << (quant_key ? "true" : "false") << ",\n";
    out << "  \"quant_value\": " << (quant_value ? "true" : "false") << ",\n";
    out << "  \"flash_attention\": " << (flash_attention ? "true" : "false") << ",\n";
    out << "  \"stage\": \"" << stage << "\",\n";
    out << "  \"run_id\": \"" << run_id << "\",\n";
    out << "  \"k_file\": \"" << k_file << "\",\n";
    out << "  \"v_file\": \"" << v_file << "\"\n";
    out << "}\n";
}

inline void WritePackedMetaFile(const std::string& path,
                                int layer_id,
                                int used_seq_len,
                                int max_seq_len,
                                int kv_heads,
                                int head_dim,
                                int bytes_per_elem,
                                int pack_h,
                                int pack_l,
                                int flash_upper_kv,
                                size_t key_stride_bytes,
                                size_t value_stride_bytes,
                                size_t key_used_bytes,
                                size_t value_used_bytes,
                                const std::string& stage,
                                const std::string& run_id,
                                const std::string& k_file,
                                const std::string& v_file) {
    std::ofstream out(path.c_str(), std::ios::out);
    if (!out) {
        return;
    }
    out << "{\n";
    out << "  \"layout\": \"packed\",\n";
    out << "  \"layer_id\": " << layer_id << ",\n";
    out << "  \"used_seq_len\": " << used_seq_len << ",\n";
    out << "  \"max_seq_len\": " << max_seq_len << ",\n";
    out << "  \"kv_heads\": " << kv_heads << ",\n";
    out << "  \"head_dim\": " << head_dim << ",\n";
    out << "  \"bytes_per_elem\": " << bytes_per_elem << ",\n";
    out << "  \"pack_h\": " << pack_h << ",\n";
    out << "  \"pack_l\": " << pack_l << ",\n";
    out << "  \"flash_upper_kv\": " << flash_upper_kv << ",\n";
    out << "  \"key_stride_bytes\": " << key_stride_bytes << ",\n";
    out << "  \"value_stride_bytes\": " << value_stride_bytes << ",\n";
    out << "  \"key_used_bytes\": " << key_used_bytes << ",\n";
    out << "  \"value_used_bytes\": " << value_used_bytes << ",\n";
    out << "  \"stage\": \"" << stage << "\",\n";
    out << "  \"run_id\": \"" << run_id << "\",\n";
    out << "  \"k_file\": \"" << k_file << "\",\n";
    out << "  \"v_file\": \"" << v_file << "\"\n";
    out << "}\n";
}

inline void DumpKVIfEnabled(const void* layer_ptr,
                            const Tensor* key,
                            const Tensor* value,
                            int add,
                            int kv_heads,
                            int head_dim,
                            int bytes_per_elem,
                            int past_length,
                            bool quant_key,
                            bool quant_value,
                            bool flash_attention,
                            const KVMeta* meta) {
    auto& cfg = GetConfig();
    if (!cfg.enabled || key == nullptr || value == nullptr) {
        return;
    }
    if (add <= 0) {
        return;
    }
    if (bytes_per_elem != 2 && bytes_per_elem != 4) {
        return;
    }
    if (!ShouldDump(cfg.stage, add)) {
        return;
    }

    int key_seq_len = 0;
    int value_seq_len = 0;
    if (kv_heads > 0 && head_dim > 0) {
        size_t denom = static_cast<size_t>(kv_heads) * static_cast<size_t>(head_dim);
        if (denom > 0) {
            size_t key_elements = key->elementSize();
            size_t value_elements = value->elementSize();
            if (key_elements % denom == 0) {
                key_seq_len = static_cast<int>(key_elements / denom);
            }
            if (value_elements % denom == 0) {
                value_seq_len = static_cast<int>(value_elements / denom);
            }
        }
    }
    if (key_seq_len <= 0) {
        key_seq_len = static_cast<int>(key->length(0));
    }
    if (value_seq_len <= 0) {
        value_seq_len = static_cast<int>(value->length(0));
    }
    int dump_seq_len = std::min(add, std::min(key_seq_len, value_seq_len));
    if (cfg.max_tokens > 0 && dump_seq_len > cfg.max_tokens) {
        dump_seq_len = cfg.max_tokens;
    }
    if (dump_seq_len <= 0) {
        return;
    }

    int layer_id = GetLayerId(layer_ptr, meta ? meta->layer_nums : 0);
    std::string stage = (add == 1 ? "decode" : "prefill");

    std::string run_dir = MNNFilePathConcat(cfg.root_dir, cfg.run_id);
    if (!EnsureDirRecursive(cfg.root_dir) || !EnsureDirRecursive(run_dir)) {
        return;
    }
    std::string layer_dir = MNNFilePathConcat(run_dir, "layer_" + std::to_string(layer_id));
    if (!EnsureDirRecursive(layer_dir)) {
        return;
    }

    int seq_start = past_length;
    int seq_end = past_length + dump_seq_len;
    std::string base = "t" + std::to_string(seq_start) + "_" + std::to_string(seq_end);
    std::string k_file = "k_" + base + ".bin";
    std::string v_file = "v_" + base + ".bin";
    std::string meta_file = "meta_" + base + ".json";

    const uint8_t* key_ptr = reinterpret_cast<const uint8_t*>(key->host<uint8_t>());
    const uint8_t* value_ptr = reinterpret_cast<const uint8_t*>(value->host<uint8_t>());
    size_t elem_count = static_cast<size_t>(dump_seq_len) * static_cast<size_t>(kv_heads) * static_cast<size_t>(head_dim);
    size_t byte_size = elem_count * static_cast<size_t>(bytes_per_elem);

    WriteBinaryFile(MNNFilePathConcat(layer_dir, k_file), key_ptr, byte_size);
    WriteBinaryFile(MNNFilePathConcat(layer_dir, v_file), value_ptr, byte_size);
    WriteMetaFile(MNNFilePathConcat(layer_dir, meta_file),
                  layer_id,
                  seq_start,
                  dump_seq_len,
                  kv_heads,
                  head_dim,
                  bytes_per_elem,
                  add,
                  key_seq_len,
                  quant_key,
                  quant_value,
                  flash_attention,
                  stage,
                  cfg.run_id,
                  k_file,
                  v_file);
}

inline void DumpPackedKVIfEnabled(const void* layer_ptr,
                                  const uint8_t* key_addr,
                                  const uint8_t* value_addr,
                                  int add,
                                  int kv_heads,
                                  int head_dim,
                                  int bytes_per_elem,
                                  int past_length,
                                  int max_length,
                                  int pack_h,
                                  int pack_l,
                                  int flash_upper_kv,
                                  size_t key_stride_bytes,
                                  size_t value_stride_bytes,
                                  bool quant_key,
                                  bool quant_value,
                                  const KVMeta* meta) {
    auto& cfg = GetConfig();
    if (!cfg.enabled || !cfg.dump_packed) {
        return;
    }
    if (key_addr == nullptr || value_addr == nullptr) {
        return;
    }
    if (add <= 0 || !ShouldDump(cfg.stage, add)) {
        return;
    }
    if (quant_key || quant_value) {
        return;
    }
    if (bytes_per_elem != 2 && bytes_per_elem != 4) {
        return;
    }
    if (kv_heads <= 0 || head_dim <= 0 || past_length <= 0) {
        return;
    }

    if (flash_upper_kv <= 0) {
        flash_upper_kv = max_length > 0 ? max_length : past_length;
    }

    size_t key_used_bytes_per_head = static_cast<size_t>(ROUND_UP(past_length, pack_h)) *
                                     static_cast<size_t>(ROUND_UP(head_dim, pack_l)) *
                                     static_cast<size_t>(bytes_per_elem);
    size_t value_used_bytes_per_head = static_cast<size_t>(UP_DIV(past_length, flash_upper_kv)) *
                                       static_cast<size_t>(ROUND_UP(head_dim, pack_h)) *
                                       static_cast<size_t>(ROUND_UP(flash_upper_kv, pack_l)) *
                                       static_cast<size_t>(bytes_per_elem);
    if (key_used_bytes_per_head > key_stride_bytes) {
        key_used_bytes_per_head = key_stride_bytes;
    }
    if (value_used_bytes_per_head > value_stride_bytes) {
        value_used_bytes_per_head = value_stride_bytes;
    }
    size_t total_key_bytes = key_used_bytes_per_head * static_cast<size_t>(kv_heads);
    size_t total_value_bytes = value_used_bytes_per_head * static_cast<size_t>(kv_heads);

    int layer_id = GetLayerId(layer_ptr, meta ? meta->layer_nums : 0);
    std::string stage = (add == 1 ? "decode" : "prefill");
    std::string run_dir = MNNFilePathConcat(cfg.root_dir, cfg.run_id);
    if (!EnsureDirRecursive(cfg.root_dir) || !EnsureDirRecursive(run_dir)) {
        return;
    }
    std::string layer_dir = MNNFilePathConcat(run_dir, "layer_" + std::to_string(layer_id));
    if (!EnsureDirRecursive(layer_dir)) {
        return;
    }

    std::string base = "packed_t0_" + std::to_string(past_length);
    std::string k_file = "packed_k_" + base + ".bin";
    std::string v_file = "packed_v_" + base + ".bin";
    std::string meta_file = "packed_meta_" + base + ".json";

    std::vector<uint8_t> key_buf(total_key_bytes);
    std::vector<uint8_t> value_buf(total_value_bytes);
    for (int h = 0; h < kv_heads; ++h) {
        std::memcpy(key_buf.data() + static_cast<size_t>(h) * key_used_bytes_per_head,
                    key_addr + static_cast<size_t>(h) * key_stride_bytes,
                    key_used_bytes_per_head);
        std::memcpy(value_buf.data() + static_cast<size_t>(h) * value_used_bytes_per_head,
                    value_addr + static_cast<size_t>(h) * value_stride_bytes,
                    value_used_bytes_per_head);
    }
    WriteBinaryFile(MNNFilePathConcat(layer_dir, k_file), key_buf.data(), key_buf.size());
    WriteBinaryFile(MNNFilePathConcat(layer_dir, v_file), value_buf.data(), value_buf.size());
    WritePackedMetaFile(MNNFilePathConcat(layer_dir, meta_file),
                        layer_id,
                        past_length,
                        max_length,
                        kv_heads,
                        head_dim,
                        bytes_per_elem,
                        pack_h,
                        pack_l,
                        flash_upper_kv,
                        key_stride_bytes,
                        value_stride_bytes,
                        key_used_bytes_per_head,
                        value_used_bytes_per_head,
                        stage,
                        cfg.run_id,
                        k_file,
                        v_file);
}

} // namespace KVExp
} // namespace MNN

#endif // MNN_EXP_DISTRIBUTION_KV_DUMP_HPP
