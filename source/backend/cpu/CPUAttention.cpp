//
//  CPUAttention.cpp
//  MNN
//
//  Created by MNN on 2024/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <limits>
#if !defined(_WIN32)
#include <dlfcn.h>
#endif
#include "CPUAttention.hpp"
#include "CPUBackend.hpp"
#include "compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/BufferAllocator.hpp"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/BufferAllocator.hpp"
#include "compute/ConvolutionTiledExecutor.hpp"

#if defined (__aarch64__)
#define FLOAT16_T __fp16
#else
#define FLOAT16_T float
#endif


namespace MNN {

template <typename T>
static void _maskQK(float * qkPacked, const float* scale, size_t seqLen, size_t processedKvSeq, int pack, int kvSeqLen, int kvoffset, int padKvSeqLen, const float* sinksPtr, const Tensor* mask, bool quantKey, bool isLowerTriangular) {
    /*
     * FIGURE 1: mask->elementSize() == seqLen * maskStride
     * Context: Cross Attention or Prefill stage (Full Context).
     * Logic:   gapLen = 0. The mask tensor dimensions match the logical QK matrix exactly.
     *          Direct access: mask[row * stride + col]
     * Row\Col   0   1   2   3
     *
     *   0       0   X   X   X    (Can only see Col 0)
     *
     *   1       0   0   X   X    (Can see Col 0, 1)
     *
     *   2       0   0   0   X    (Can see Col 0, 1, 2)
     *
     *   3       0   0   0   0    (Fully visible)
     *
     * Legend:
     *   '0' : Visible (Value = Scale * QK)
     *   'X' : Masked  (Value = -inf)
     */


    /*
     * FIGURE 2: mask->elementSize() != seqLen * maskStride
     * Context: Self-Attention Inference (Decoding stage).
     * Logic:   gapLen = maskStride - seqLen (Right Alignment).
     *          The "Gap" represents History KV Cache, which is implicitly visible.
     *          The Mask Tensor only covers the current sequence window.
     *
     * Example: maskStride (Total KV) = 6
     *          seqLen (Current Q)    = 4
     *          gapLen                = 6 - 4 = 2
     *
     * Structure:
     *   - Cols [0, 1]: "Gap" / History region. Code logic: `if (col < gapLen) continue;`.
     *                  No mask is added, so they remain Visible ('0').
     *   - Cols [2-5]:  "Current" region. Code logic: `mask[col - gapLen]`.
     *
     * Row\Col   0   1   |   2   3   4   5
     *          (Gap)    |   (Mask Tensor Region)
     *
     *   0       0   0   |   0   X   X   X    <-- Mask row 0 applies to Col 2~5
     *                   |
     *   1       0   0   |   0   0   X   X    <-- Mask row 1 applies to Col 2~5
     *                   |
     *   2       0   0   |   0   0   0   X    <-- Mask row 2 applies to Col 2~5
     *                   |
     *   3       0   0   |   0   0   0   0    <-- Mask row 3 applies to Col 2~5
     *
     * Legend:
     *   '0' (Left)  : History KV, implicitly visible (code skips mask addition).
     *   '0' (Right) : Current KV, visible according to Mask Tensor.
     *   'X'         : Masked by Mask Tensor (-inf).
     */

    if (isLowerTriangular && quantKey) {
        return;
    }
    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
    auto source = (T*)qkPacked;
    float scaleVal = scale[0];
    int gapLen = (mask->elementSize() == (seqLen + padKvSeqLen) * (kvSeqLen + padKvSeqLen)) ? 0 : static_cast<int>(kvSeqLen - seqLen);

    auto kvBlockCount = UP_DIV(processedKvSeq, pack);
    auto qkSize = ROUND_UP(processedKvSeq, pack) * seqLen;

    if (isLowerTriangular) {
        for (int i = 0; i < qkSize; ++i) {
            source[i] *= scaleVal;
        }
        return;
    }

    if (mask == nullptr) {
        return;
    }

    auto maskPtr = mask->host<T>();

    // not lower triangular
    auto maskCols = (mask->elementSize() == (seqLen + padKvSeqLen) * (kvSeqLen + padKvSeqLen)) ? kvSeqLen + padKvSeqLen : seqLen + padKvSeqLen;
    for (int i = 0; i < kvBlockCount; ++i) {
        T* blockDataPtr = source + (i * seqLen * pack);

        for (int j = 0; j < seqLen; ++j) {
            T* dataPtr = blockDataPtr + (j * pack);
            const T* currentMaskRow = maskPtr + j * maskCols;

            for (int k = 0; k < pack; ++k) {
                float val = (float)dataPtr[k];
                if (!quantKey) {
                    val *= scaleVal;
                    dataPtr[k] = (T)val;
                }
                int currentKvSeqIndx = kvoffset + i * pack + k; // kvoffset=i*mBlockKv

                if (currentKvSeqIndx < gapLen) {
                    continue;
                }
                if (currentKvSeqIndx - gapLen >= maskCols) {
                    break;
                }

                val += (float)currentMaskRow[currentKvSeqIndx - gapLen];
                dataPtr[k] = (T)val;

            }
        }
    }
}

static inline int clampInt(int value, int low, int high) {
    return ALIMAX(low, ALIMIN(high, value));
}

struct PredictorFlags {
    bool raw = true;
    bool deltaSeq = false;
    bool xorSeq = false;
};

struct ZstdApi {
    bool initialized = false;
    bool available = false;
#if !defined(_WIN32)
    void* handle = nullptr;
    size_t (*compressBound)(size_t) = nullptr;
    size_t (*compress)(void*, size_t, const void*, size_t, int) = nullptr;
    size_t (*decompress)(void*, size_t, const void*, size_t) = nullptr;
    unsigned (*isError)(size_t) = nullptr;
#endif
};

static ZstdApi& getZstdApi() {
    static ZstdApi api;
    if (api.initialized) {
        return api;
    }
    api.initialized = true;
#if !defined(_WIN32)
    const char* libs[] = {"libzstd.so.1", "libzstd.so", "libzstd.dylib"};
    for (const auto* lib : libs) {
        api.handle = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);
        if (api.handle != nullptr) {
            break;
        }
    }
    if (api.handle != nullptr) {
        api.compressBound = reinterpret_cast<size_t(*)(size_t)>(dlsym(api.handle, "ZSTD_compressBound"));
        api.compress = reinterpret_cast<size_t(*)(void*, size_t, const void*, size_t, int)>(dlsym(api.handle, "ZSTD_compress"));
        api.decompress = reinterpret_cast<size_t(*)(void*, size_t, const void*, size_t)>(dlsym(api.handle, "ZSTD_decompress"));
        api.isError = reinterpret_cast<unsigned(*)(size_t)>(dlsym(api.handle, "ZSTD_isError"));
        api.available = (api.compressBound != nullptr
            && api.compress != nullptr
            && api.decompress != nullptr
            && api.isError != nullptr);
    }
#endif
    return api;
}

static inline void writeLe32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xFFu));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFFu));
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFFu));
}

static inline bool readLe32(const uint8_t* data, size_t size, size_t& off, uint32_t& out) {
    if (data == nullptr || off + 4 > size) {
        return false;
    }
    out = static_cast<uint32_t>(data[off + 0])
        | (static_cast<uint32_t>(data[off + 1]) << 8)
        | (static_cast<uint32_t>(data[off + 2]) << 16)
        | (static_cast<uint32_t>(data[off + 3]) << 24);
    off += 4;
    return true;
}

static uint64_t fnv1a64(const uint8_t* data, size_t size, uint64_t seed = 1469598103934665603ULL) {
    uint64_t hash = seed;
    if (data == nullptr || size == 0) {
        return hash;
    }
    constexpr uint64_t kPrime = 1099511628211ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= kPrime;
    }
    return hash;
}

static bool zstdCompressToBuffer(const uint8_t* data,
                                 size_t size,
                                 int level,
                                 std::vector<uint8_t>& out) {
    out.clear();
    if (data == nullptr || size == 0) {
        return false;
    }
    auto& api = getZstdApi();
    if (!api.available) {
        return false;
    }
#if !defined(_WIN32)
    const size_t bound = api.compressBound(size);
    if (bound == 0 || bound < size / 2) {
        return false;
    }
    out.resize(bound);
    const size_t ret = api.compress(out.data(), out.size(), data, size, level);
    if (api.isError(ret) != 0) {
        out.clear();
        return false;
    }
    out.resize(ret);
    return true;
#else
    (void)level;
    return false;
#endif
}

static bool zstdDecompressToBuffer(const uint8_t* data,
                                   size_t size,
                                   size_t expectedBytes,
                                   std::vector<uint8_t>& out) {
    out.clear();
    if (data == nullptr || size == 0 || expectedBytes == 0) {
        return false;
    }
    auto& api = getZstdApi();
    if (!api.available) {
        return false;
    }
#if !defined(_WIN32)
    out.resize(expectedBytes);
    const size_t ret = api.decompress(out.data(), out.size(), data, size);
    if (api.isError(ret) != 0 || ret != expectedBytes) {
        out.clear();
        return false;
    }
    return true;
#else
    return false;
#endif
}

static PredictorFlags parsePredictorFlags(const std::string& spec, bool defaultRaw) {
    PredictorFlags flags;
    flags.raw = defaultRaw;
    if (spec.empty()) {
        return flags;
    }
    flags.raw = false;
    size_t begin = 0;
    while (begin < spec.size()) {
        size_t end = spec.find(',', begin);
        if (end == std::string::npos) {
            end = spec.size();
        }
        size_t l = begin;
        while (l < end && (spec[l] == ' ' || spec[l] == '\t')) {
            ++l;
        }
        size_t r = end;
        while (r > l && (spec[r - 1] == ' ' || spec[r - 1] == '\t')) {
            --r;
        }
        const auto token = spec.substr(l, r - l);
        if (token == "raw") {
            flags.raw = true;
        } else if (token == "delta_seq") {
            flags.deltaSeq = true;
        } else if (token == "xor_seq") {
            flags.xorSeq = true;
        }
        begin = end + 1;
    }
    if (!flags.raw && !flags.deltaSeq && !flags.xorSeq) {
        flags.raw = true;
    }
    return flags;
}

static void buildDeltaSeq(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst) {
    dst.resize(src.size());
    uint8_t prev = 0;
    for (size_t i = 0; i < src.size(); ++i) {
        const uint8_t cur = src[i];
        dst[i] = static_cast<uint8_t>(cur - prev);
        prev = cur;
    }
}

static void buildXorSeq(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst) {
    dst.resize(src.size());
    uint8_t prev = 0;
    for (size_t i = 0; i < src.size(); ++i) {
        const uint8_t cur = src[i];
        dst[i] = static_cast<uint8_t>(cur ^ prev);
        prev = cur;
    }
}

static void recoverDeltaSeq(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst) {
    dst.resize(src.size());
    uint8_t prev = 0;
    for (size_t i = 0; i < src.size(); ++i) {
        prev = static_cast<uint8_t>(prev + src[i]);
        dst[i] = prev;
    }
}

static void recoverXorSeq(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst) {
    dst.resize(src.size());
    uint8_t prev = 0;
    for (size_t i = 0; i < src.size(); ++i) {
        const uint8_t cur = static_cast<uint8_t>(src[i] ^ prev);
        dst[i] = cur;
        prev = cur;
    }
}

static inline size_t repeatRunLen(const uint8_t* data, size_t n, size_t pos) {
    if (pos >= n) {
        return 0;
    }
    size_t run = 1;
    while (pos + run < n && data[pos + run] == data[pos] && run < 131) {
        ++run;
    }
    return run;
}

// RLE stream format:
// - Literal run: 1-byte ctrl [0..127] means len=ctrl+1, followed by literal bytes.
// - Repeat  run: 1-byte ctrl [128..255] means len=(ctrl&127)+4, followed by repeated byte.
static bool rleEncode(const uint8_t* data, size_t size, std::vector<uint8_t>& out) {
    out.clear();
    if (data == nullptr || size == 0) {
        return false;
    }
    out.reserve(size + size / 8 + 16);
    size_t i = 0;
    while (i < size) {
        size_t run = repeatRunLen(data, size, i);
        if (run >= 4) {
            MNN_ASSERT(run >= 4 && run <= 131);
            out.push_back(static_cast<uint8_t>(0x80 | static_cast<uint8_t>(run - 4)));
            out.push_back(data[i]);
            i += run;
            continue;
        }
        const size_t litBegin = i;
        size_t litLen = 0;
        while (i < size && litLen < 128) {
            run = repeatRunLen(data, size, i);
            if (run >= 4) {
                break;
            }
            ++i;
            ++litLen;
        }
        if (litLen == 0) {
            return false;
        }
        out.push_back(static_cast<uint8_t>(litLen - 1));
        out.insert(out.end(), data + litBegin, data + litBegin + litLen);
    }
    return true;
}

static bool rleDecode(const uint8_t* data, size_t size, size_t expectedBytes, std::vector<uint8_t>& out) {
    out.clear();
    if (data == nullptr || size == 0 || expectedBytes == 0) {
        return false;
    }
    out.reserve(expectedBytes);
    size_t off = 0;
    while (off < size) {
        const uint8_t ctrl = data[off++];
        if (ctrl < 128) {
            const size_t litLen = static_cast<size_t>(ctrl) + 1;
            if (off + litLen > size || out.size() + litLen > expectedBytes) {
                return false;
            }
            out.insert(out.end(), data + off, data + off + litLen);
            off += litLen;
            continue;
        }
        const size_t runLen = static_cast<size_t>(ctrl & 127) + 4;
        if (off >= size || out.size() + runLen > expectedBytes) {
            return false;
        }
        out.insert(out.end(), runLen, data[off]);
        off += 1;
    }
    return out.size() == expectedBytes;
}

enum PredictiveMode : uint8_t {
    MODE_RAW = 0,
    MODE_DELTA_SEQ = 1,
    MODE_XOR_SEQ = 2,
};

enum PayloadCodec : uint8_t {
    CODEC_RLE = 0,
    CODEC_ZSTD = 1,
};

static bool encodePredictiveStream(const std::vector<uint8_t>& src,
                                   const PredictorFlags& flags,
                                   std::vector<uint8_t>& out) {
    out.clear();
    if (src.empty()) {
        out.reserve(10);
        out.push_back(static_cast<uint8_t>(MODE_RAW));
        out.push_back(static_cast<uint8_t>(CODEC_RLE));
        writeLe32(out, 0);
        writeLe32(out, 0);
        return true;
    }

    struct Candidate {
        bool valid = false;
        uint8_t mode = static_cast<uint8_t>(MODE_RAW);
        uint8_t codec = static_cast<uint8_t>(CODEC_RLE);
        std::vector<uint8_t> payload;
    };
    Candidate best;
    std::vector<uint8_t> transformed;
    std::vector<uint8_t> rlePayload;
    std::vector<uint8_t> zstdPayload;

    auto tryMode = [&](uint8_t mode, const std::vector<uint8_t>& stream) {
        if (!rleEncode(stream.data(), stream.size(), rlePayload)) {
            return;
        }
        Candidate rleCand;
        rleCand.valid = true;
        rleCand.mode = mode;
        rleCand.codec = static_cast<uint8_t>(CODEC_RLE);
        rleCand.payload = rlePayload;

        Candidate zstdCand;
        if (zstdCompressToBuffer(stream.data(), stream.size(), 3, zstdPayload) && !zstdPayload.empty()) {
            zstdCand.valid = true;
            zstdCand.mode = mode;
            zstdCand.codec = static_cast<uint8_t>(CODEC_ZSTD);
            zstdCand.payload = zstdPayload;
        }

        auto better = [](const Candidate& a, const Candidate& b) {
            if (!b.valid) {
                return true;
            }
            return a.payload.size() < b.payload.size();
        };
        if (rleCand.valid && better(rleCand, best)) {
            best = std::move(rleCand);
        }
        if (zstdCand.valid && better(zstdCand, best)) {
            best = std::move(zstdCand);
        }
    };

    if (flags.raw) {
        tryMode(static_cast<uint8_t>(MODE_RAW), src);
    }
    if (flags.deltaSeq) {
        buildDeltaSeq(src, transformed);
        tryMode(static_cast<uint8_t>(MODE_DELTA_SEQ), transformed);
    }
    if (flags.xorSeq) {
        buildXorSeq(src, transformed);
        tryMode(static_cast<uint8_t>(MODE_XOR_SEQ), transformed);
    }
    if (!best.valid) {
        tryMode(static_cast<uint8_t>(MODE_RAW), src);
    }
    if (!best.valid) {
        return false;
    }

    out.reserve(10 + best.payload.size());
    out.push_back(best.mode);
    out.push_back(best.codec);
    writeLe32(out, static_cast<uint32_t>(src.size()));
    writeLe32(out, static_cast<uint32_t>(best.payload.size()));
    out.insert(out.end(), best.payload.begin(), best.payload.end());
    return true;
}

static bool decodePredictiveStream(const uint8_t* data,
                                   size_t size,
                                   size_t& off,
                                   std::vector<uint8_t>& out) {
    out.clear();
    if (data == nullptr || off + 10 > size) {
        return false;
    }
    const uint8_t mode = data[off++];
    const uint8_t codec = data[off++];
    uint32_t rawLen = 0;
    uint32_t payloadLen = 0;
    if (!readLe32(data, size, off, rawLen) || !readLe32(data, size, off, payloadLen)) {
        return false;
    }
    if (off + payloadLen > size) {
        return false;
    }
    const uint8_t* payloadPtr = data + off;
    off += payloadLen;
    if (rawLen == 0) {
        out.clear();
        return payloadLen == 0;
    }

    std::vector<uint8_t> transformed;
    if (codec == static_cast<uint8_t>(CODEC_RLE)) {
        if (!rleDecode(payloadPtr, payloadLen, rawLen, transformed)) {
            return false;
        }
    } else if (codec == static_cast<uint8_t>(CODEC_ZSTD)) {
        if (!zstdDecompressToBuffer(payloadPtr, payloadLen, rawLen, transformed)) {
            return false;
        }
    } else {
        return false;
    }
    if (transformed.size() != rawLen) {
        return false;
    }

    if (mode == static_cast<uint8_t>(MODE_RAW)) {
        out = std::move(transformed);
    } else if (mode == static_cast<uint8_t>(MODE_DELTA_SEQ)) {
        recoverDeltaSeq(transformed, out);
    } else if (mode == static_cast<uint8_t>(MODE_XOR_SEQ)) {
        recoverXorSeq(transformed, out);
    } else {
        return false;
    }
    return out.size() == rawLen;
}

static bool encodeFp16GearPredictive(const uint8_t* data,
                                     size_t bytes,
                                     const PredictorFlags& loFlags,
                                     const PredictorFlags& hiFlags,
                                     std::vector<uint8_t>& out) {
    out.clear();
    if (data == nullptr || bytes < 2 || (bytes % 2) != 0) {
        return false;
    }
    const size_t words = bytes / 2;
    std::vector<uint8_t> lo(words);
    std::vector<uint8_t> hi(words);
    for (size_t i = 0; i < words; ++i) {
        lo[i] = data[2 * i + 0];
        hi[i] = data[2 * i + 1];
    }
    std::vector<uint8_t> loFrame;
    std::vector<uint8_t> hiFrame;
    if (!encodePredictiveStream(lo, loFlags, loFrame)
        || !encodePredictiveStream(hi, hiFlags, hiFrame)) {
        return false;
    }

    out.reserve(4 + loFrame.size() + hiFrame.size());
    writeLe32(out, static_cast<uint32_t>(words));
    out.insert(out.end(), loFrame.begin(), loFrame.end());
    out.insert(out.end(), hiFrame.begin(), hiFrame.end());
    return true;
}

static bool decodeFp16GearPredictive(const uint8_t* data, size_t size, std::vector<uint8_t>& out) {
    out.clear();
    if (data == nullptr || size < 4) {
        return false;
    }
    size_t off = 0;
    uint32_t words = 0;
    if (!readLe32(data, size, off, words) || words == 0) {
        return false;
    }

    std::vector<uint8_t> lo;
    std::vector<uint8_t> hi;
    if (!decodePredictiveStream(data, size, off, lo)
        || !decodePredictiveStream(data, size, off, hi)) {
        return false;
    }
    if (lo.size() != words || hi.size() != words || off != size) {
        return false;
    }

    out.resize(static_cast<size_t>(words) * 2);
    for (size_t i = 0; i < words; ++i) {
        out[2 * i + 0] = lo[i];
        out[2 * i + 1] = hi[i];
    }
    return true;
}

static bool encodeFp32LanePredictive(const uint8_t* data,
                                     size_t bytes,
                                     const PredictorFlags& flags,
                                     std::vector<uint8_t>& out) {
    out.clear();
    if (data == nullptr || bytes < 4 || (bytes % 4) != 0) {
        return false;
    }
    const size_t words = bytes / 4;
    std::vector<uint8_t> lane0(words), lane1(words), lane2(words), lane3(words);
    for (size_t i = 0; i < words; ++i) {
        lane0[i] = data[4 * i + 0];
        lane1[i] = data[4 * i + 1];
        lane2[i] = data[4 * i + 2];
        lane3[i] = data[4 * i + 3];
    }
    std::vector<uint8_t> f0, f1, f2, f3;
    if (!encodePredictiveStream(lane0, flags, f0)
        || !encodePredictiveStream(lane1, flags, f1)
        || !encodePredictiveStream(lane2, flags, f2)
        || !encodePredictiveStream(lane3, flags, f3)) {
        return false;
    }
    out.reserve(4 + f0.size() + f1.size() + f2.size() + f3.size());
    writeLe32(out, static_cast<uint32_t>(words));
    out.insert(out.end(), f0.begin(), f0.end());
    out.insert(out.end(), f1.begin(), f1.end());
    out.insert(out.end(), f2.begin(), f2.end());
    out.insert(out.end(), f3.begin(), f3.end());
    return true;
}

static bool decodeFp32LanePredictive(const uint8_t* data, size_t size, std::vector<uint8_t>& out) {
    out.clear();
    if (data == nullptr || size < 4) {
        return false;
    }
    size_t off = 0;
    uint32_t words = 0;
    if (!readLe32(data, size, off, words) || words == 0) {
        return false;
    }

    std::vector<uint8_t> lanes[4];
    for (int i = 0; i < 4; ++i) {
        if (!decodePredictiveStream(data, size, off, lanes[i])) {
            return false;
        }
        if (lanes[i].size() != words) {
            return false;
        }
    }
    if (off != size) {
        return false;
    }

    out.resize(static_cast<size_t>(words) * 4);
    for (size_t i = 0; i < words; ++i) {
        out[4 * i + 0] = lanes[0][i];
        out[4 * i + 1] = lanes[1][i];
        out[4 * i + 2] = lanes[2][i];
        out[4 * i + 3] = lanes[3][i];
    }
    return true;
}

static bool collectKvMergedRange(CPUKVCacheManager* cacheManager,
                                 int kvNumHead,
                                 int headDim,
                                 int bytes,
                                 int lPack,
                                 int hPack,
                                 int flashBlockKv,
                                 int startToken,
                                 int tokenCount,
                                 std::vector<uint8_t>& keyMerged,
                                 std::vector<uint8_t>& valueMerged) {
    keyMerged.clear();
    valueMerged.clear();
    if (cacheManager == nullptr
        || tokenCount <= 0
        || kvNumHead <= 0
        || headDim <= 0
        || bytes <= 0
        || startToken < 0) {
        return false;
    }
    const int rangeEnd = startToken + tokenCount;
    const int activeLen = cacheManager->kvLength();
    if (rangeEnd < startToken || rangeEnd > cacheManager->maxLength() || rangeEnd > activeLen) {
        return false;
    }

    const int safeFlashBlockKv = ALIMAX(1, flashBlockKv);
    std::vector<const uint8_t*> keyPtrs;
    std::vector<const uint8_t*> valuePtrs;
    keyPtrs.reserve(kvNumHead);
    valuePtrs.reserve(kvNumHead);
    for (int h = 0; h < kvNumHead; ++h) {
        const uint8_t* keyPtr = reinterpret_cast<const uint8_t*>(cacheManager->addrOfKey(h));
        const uint8_t* valuePtr = reinterpret_cast<const uint8_t*>(cacheManager->addrOfValue(h));
        if (keyPtr == nullptr || valuePtr == nullptr) {
            return false;
        }
        keyPtrs.emplace_back(keyPtr);
        valuePtrs.emplace_back(valuePtr);
    }

    const size_t logicalBytesPerHead = (size_t)tokenCount * (size_t)headDim * (size_t)bytes;
    const size_t totalBytes = (size_t)keyPtrs.size() * logicalBytesPerHead;
    keyMerged.resize(totalBytes);
    valueMerged.resize(totalBytes);
    uint8_t* keyOut = keyMerged.data();
    uint8_t* valueOut = valueMerged.data();

    const size_t keyStride0 = (size_t)ROUND_UP(headDim, lPack) * (size_t)hPack;
    const size_t keyStride1 = (size_t)hPack * (size_t)lPack;
    const size_t valueStride2 = (size_t)lPack * (size_t)hPack;
    const size_t valueStride1 = (size_t)UP_DIV(safeFlashBlockKv, lPack) * valueStride2;
    const size_t valueStride0 = valueStride1 * (size_t)UP_DIV(headDim, hPack);
    const size_t keyCapacity = cacheManager->keySizePerHead() / (size_t)bytes;
    const size_t valueCapacity = cacheManager->valueSizePerHead() / (size_t)bytes;
    if (keyCapacity == 0 || valueCapacity == 0) {
        return false;
    }

    for (size_t h = 0; h < keyPtrs.size(); ++h) {
        const uint8_t* keyPtr = keyPtrs[h];
        const uint8_t* valuePtr = valuePtrs[h];
        for (int local = 0; local < tokenCount; ++local) {
            const int seq = startToken + local;
            const int seqOut = seq / hPack;
            const int seqIn = seq % hPack;
            const size_t keySeqBase = (size_t)seqOut * keyStride0 + (size_t)seqIn * (size_t)lPack;

            const int seqInFlash = seq % safeFlashBlockKv;
            const size_t valueInner = (size_t)(seq / safeFlashBlockKv) * valueStride0
                + (size_t)(seqInFlash / lPack) * valueStride2
                + (size_t)(seqInFlash % lPack);

            for (int dim = 0; dim < headDim; ++dim) {
                const size_t keyIndex = keySeqBase
                    + (size_t)(dim / lPack) * keyStride1
                    + (size_t)(dim % lPack);
                if (keyIndex >= keyCapacity) {
                    return false;
                }
                const uint8_t* keyElem = keyPtr + keyIndex * (size_t)bytes;
                ::memcpy(keyOut, keyElem, (size_t)bytes);
                keyOut += bytes;

                const size_t valueIndex = (size_t)(dim / hPack) * valueStride1
                    + (size_t)(dim % hPack) * (size_t)lPack
                    + valueInner;
                if (valueIndex >= valueCapacity) {
                    return false;
                }
                const uint8_t* valueElem = valuePtr + valueIndex * (size_t)bytes;
                ::memcpy(valueOut, valueElem, (size_t)bytes);
                valueOut += bytes;
            }
        }
    }
    return true;
}

static bool scatterKvMergedRange(CPUKVCacheManager* cacheManager,
                                 int kvNumHead,
                                 int headDim,
                                 int bytes,
                                 int lPack,
                                 int hPack,
                                 int flashBlockKv,
                                 int startToken,
                                 int tokenCount,
                                 const std::vector<uint8_t>& keyMerged,
                                 const std::vector<uint8_t>& valueMerged) {
    if (cacheManager == nullptr
        || tokenCount <= 0
        || kvNumHead <= 0
        || headDim <= 0
        || bytes <= 0
        || startToken < 0) {
        return false;
    }
    const int rangeEnd = startToken + tokenCount;
    const int activeLen = cacheManager->kvLength();
    if (rangeEnd < startToken || rangeEnd > cacheManager->maxLength() || rangeEnd > activeLen) {
        return false;
    }
    const int safeFlashBlockKv = ALIMAX(1, flashBlockKv);
    const size_t logicalBytesPerHead = (size_t)tokenCount * (size_t)headDim * (size_t)bytes;
    const size_t expectedBytes = (size_t)kvNumHead * logicalBytesPerHead;
    if (keyMerged.size() != expectedBytes || valueMerged.size() != expectedBytes) {
        return false;
    }

    std::vector<uint8_t*> keyPtrs;
    std::vector<uint8_t*> valuePtrs;
    keyPtrs.reserve(kvNumHead);
    valuePtrs.reserve(kvNumHead);
    for (int h = 0; h < kvNumHead; ++h) {
        auto* keyPtr = reinterpret_cast<uint8_t*>(cacheManager->addrOfKey(h));
        auto* valuePtr = reinterpret_cast<uint8_t*>(cacheManager->addrOfValue(h));
        if (keyPtr == nullptr || valuePtr == nullptr) {
            return false;
        }
        keyPtrs.emplace_back(keyPtr);
        valuePtrs.emplace_back(valuePtr);
    }

    const size_t keyStride0 = (size_t)ROUND_UP(headDim, lPack) * (size_t)hPack;
    const size_t keyStride1 = (size_t)hPack * (size_t)lPack;
    const size_t valueStride2 = (size_t)lPack * (size_t)hPack;
    const size_t valueStride1 = (size_t)UP_DIV(safeFlashBlockKv, lPack) * valueStride2;
    const size_t valueStride0 = valueStride1 * (size_t)UP_DIV(headDim, hPack);
    const size_t keyCapacity = cacheManager->keySizePerHead() / (size_t)bytes;
    const size_t valueCapacity = cacheManager->valueSizePerHead() / (size_t)bytes;
    if (keyCapacity == 0 || valueCapacity == 0) {
        return false;
    }

    const uint8_t* keyIn = keyMerged.data();
    const uint8_t* valueIn = valueMerged.data();
    for (size_t h = 0; h < keyPtrs.size(); ++h) {
        uint8_t* keyPtr = keyPtrs[h];
        uint8_t* valuePtr = valuePtrs[h];
        for (int local = 0; local < tokenCount; ++local) {
            const int seq = startToken + local;
            const int seqOut = seq / hPack;
            const int seqIn = seq % hPack;
            const size_t keySeqBase = (size_t)seqOut * keyStride0 + (size_t)seqIn * (size_t)lPack;

            const int seqInFlash = seq % safeFlashBlockKv;
            const size_t valueInner = (size_t)(seq / safeFlashBlockKv) * valueStride0
                + (size_t)(seqInFlash / lPack) * valueStride2
                + (size_t)(seqInFlash % lPack);

            for (int dim = 0; dim < headDim; ++dim) {
                const size_t keyIndex = keySeqBase
                    + (size_t)(dim / lPack) * keyStride1
                    + (size_t)(dim % lPack);
                if (keyIndex >= keyCapacity) {
                    return false;
                }
                uint8_t* keyElem = keyPtr + keyIndex * (size_t)bytes;
                ::memcpy(keyElem, keyIn, (size_t)bytes);
                keyIn += bytes;

                const size_t valueIndex = (size_t)(dim / hPack) * valueStride1
                    + (size_t)(dim % hPack) * (size_t)lPack
                    + valueInner;
                if (valueIndex >= valueCapacity) {
                    return false;
                }
                uint8_t* valueElem = valuePtr + valueIndex * (size_t)bytes;
                ::memcpy(valueElem, valueIn, (size_t)bytes);
                valueIn += bytes;
            }
        }
    }
    return true;
}

static bool zeroKvRange(CPUKVCacheManager* cacheManager,
                        int kvNumHead,
                        int headDim,
                        int bytes,
                        int lPack,
                        int hPack,
                        int flashBlockKv,
                        int startToken,
                        int tokenCount) {
    if (cacheManager == nullptr
        || tokenCount <= 0
        || kvNumHead <= 0
        || headDim <= 0
        || bytes <= 0
        || startToken < 0) {
        return false;
    }
    const int rangeEnd = startToken + tokenCount;
    const int activeLen = cacheManager->kvLength();
    if (rangeEnd < startToken || rangeEnd > cacheManager->maxLength() || rangeEnd > activeLen) {
        return false;
    }
    const int safeFlashBlockKv = ALIMAX(1, flashBlockKv);
    const size_t keyStride0 = (size_t)ROUND_UP(headDim, lPack) * (size_t)hPack;
    const size_t keyStride1 = (size_t)hPack * (size_t)lPack;
    const size_t valueStride2 = (size_t)lPack * (size_t)hPack;
    const size_t valueStride1 = (size_t)UP_DIV(safeFlashBlockKv, lPack) * valueStride2;
    const size_t valueStride0 = valueStride1 * (size_t)UP_DIV(headDim, hPack);
    const size_t keyCapacity = cacheManager->keySizePerHead() / (size_t)bytes;
    const size_t valueCapacity = cacheManager->valueSizePerHead() / (size_t)bytes;
    if (keyCapacity == 0 || valueCapacity == 0) {
        return false;
    }
    for (int h = 0; h < kvNumHead; ++h) {
        auto* keyPtr = reinterpret_cast<uint8_t*>(cacheManager->addrOfKey(h));
        auto* valuePtr = reinterpret_cast<uint8_t*>(cacheManager->addrOfValue(h));
        if (keyPtr == nullptr || valuePtr == nullptr) {
            return false;
        }
        for (int local = 0; local < tokenCount; ++local) {
            const int seq = startToken + local;
            const int seqOut = seq / hPack;
            const int seqIn = seq % hPack;
            const size_t keySeqBase = (size_t)seqOut * keyStride0 + (size_t)seqIn * (size_t)lPack;

            const int seqInFlash = seq % safeFlashBlockKv;
            const size_t valueInner = (size_t)(seq / safeFlashBlockKv) * valueStride0
                + (size_t)(seqInFlash / lPack) * valueStride2
                + (size_t)(seqInFlash % lPack);

            for (int dim = 0; dim < headDim; ++dim) {
                const size_t keyIndex = keySeqBase
                    + (size_t)(dim / lPack) * keyStride1
                    + (size_t)(dim % lPack);
                if (keyIndex >= keyCapacity) {
                    return false;
                }
                auto* keyElem = keyPtr + keyIndex * (size_t)bytes;
                ::memset(keyElem, 0, (size_t)bytes);

                const size_t valueIndex = (size_t)(dim / hPack) * valueStride1
                    + (size_t)(dim % hPack) * (size_t)lPack
                    + valueInner;
                if (valueIndex >= valueCapacity) {
                    return false;
                }
                auto* valueElem = valuePtr + valueIndex * (size_t)bytes;
                ::memset(valueElem, 0, (size_t)bytes);
            }
        }
    }
    return true;
}

ErrorCode CPUAttention::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto gcore = static_cast<CPUBackend *>(backend())->functions();
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    gcore->MNNGetMatMulPackMode(&eP, &lP, &hP);
    mThreadNum = ((CPUBackend *)backend())->threadNumber();
    mPack  = gcore->pack;
    mBytes = gcore->bytes;
    int attentionOption = static_cast<CPUBackend *>(backend())->getRuntime()->hint().attentionOption;
    mUseFlashAttention = (attentionOption / 8 == 1);

    // If slide window attention applied, quant key/value must be diabled
    mQuantKey = inputs.size() < 5 && (attentionOption % 8 >= 1);
    mQuantValue = inputs.size() < 5 && (attentionOption % 8 > 1) && mUseFlashAttention;
    static_cast<CPUBackend*>(backend())->int8Functions()->MNNGetGemmUnit(&hP8, &lP8, &eP8);

    auto query = inputs[0];
    auto key   = inputs[1];
    int seqLen = query->length(1);
    int mBlockNum = 1;
    mNumHead = query->length(2);
    mHeadDim = query->length(3);
    mKvNumHead = key->length(2);
    mKVCacheManager->setAttenQuantKeyValue(mUseFlashAttention, mQuantKey, mQuantValue);
    mKVCacheManager->onResize(mKvNumHead, mHeadDim);

    // Common buffer allocated
    auto bufferAlloc = static_cast<CPUBackend*>(backend())->getBufferAllocator();
    mPackQKV.reset(Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(mHeadDim, mPack), seqLen, mPack * mBytes}));
    backend()->onAcquireBuffer(mPackQKV.get(), Backend::DYNAMIC);
    if (inputs.size() > 4 || mUseFlashAttention) { // needed by flash attention and sliding attention with sink
        mRunningMax.reset(Tensor::createDevice<int8_t>({mThreadNum, seqLen * 4}));
        mRunningSum.reset(Tensor::createDevice<int8_t>({mThreadNum, seqLen * 4}));
        backend()->onAcquireBuffer(mRunningMax.get(), Backend::DYNAMIC);
        backend()->onAcquireBuffer(mRunningSum.get(), Backend::DYNAMIC);
    }
    if (mUseFlashAttention) { // extra buffer need by flash attention
        mExpfDiffMax.reset(Tensor::createDevice<int8_t>({mThreadNum, seqLen * 4}));
        mTempOut.reset(Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(mHeadDim, mPack), seqLen, mPack * mBytes}));
        backend()->onAcquireBuffer(mExpfDiffMax.get(), Backend::DYNAMIC);
        backend()->onAcquireBuffer(mTempOut.get(), Backend::DYNAMIC);
    }
    if (mQuantKey) {
        int outterSeqLen = UP_DIV(seqLen, eP8);
        int outterHeadDim = UP_DIV(mHeadDim, lP8);

        size_t packedQSize = 0;
        if (outterSeqLen > 0) {
            int fullSeqBlocks = (seqLen / eP8);
            packedQSize += (size_t)fullSeqBlocks * outterHeadDim * eP8 * lP8;

            int lastEUnit = seqLen % eP8;
            if (lastEUnit != 0) {
                packedQSize += (size_t)outterHeadDim * lastEUnit * lP8;
            }
        }
        mPackQ.reset(Tensor::createDevice<int8_t>({mNumHead, (int32_t)packedQSize}));
        backend()->onAcquireBuffer(mPackQ.get(), Backend::DYNAMIC);

        mSumQ = bufferAlloc->alloc(mThreadNum * ROUND_UP(seqLen, eP8) * mBlockNum * sizeof(int32_t));
        mQueryScale = bufferAlloc->alloc(mNumHead * seqLen * mBlockNum * QUANT_INFO_BYTES);
        mQueryZeroPoint = bufferAlloc->alloc(mNumHead * seqLen * mBlockNum * QUANT_INFO_BYTES);
        mQueryQuantZero = bufferAlloc->alloc(mNumHead * seqLen * mBlockNum * QUANT_INFO_BYTES);
        mQueryQuantScale = bufferAlloc->alloc(mNumHead * seqLen * mBlockNum * QUANT_INFO_BYTES);
        mQuantQuery = bufferAlloc->alloc(seqLen * mNumHead * UP_DIV(mHeadDim, gcore->pack) * gcore->pack);

        if (mBlockNum > 1) {
            mAccumBuffer = bufferAlloc->alloc(eP8 * hP8 * mThreadNum * QUANT_INFO_BYTES);
            if (mAccumBuffer.invalid()) {
                return OUT_OF_MEMORY;
            }
        }

        if (mSumQ.invalid() || mQueryScale.invalid() || mQueryQuantZero.invalid() || mQueryZeroPoint.invalid() || mQueryQuantScale.invalid() || mQuantQuery.invalid()) {
            return OUT_OF_MEMORY;
        }

        // post parameters for int8 gemm
        mGemmRelu.reset(2 * sizeof(int32_t));
        if (!mGemmRelu.get()) {
            MNN_ERROR("Allocate mGemmRelu buffer failed in CPU Attention");
            return OUT_OF_MEMORY;
        }
        ((float*)mGemmRelu.get())[0] = -std::numeric_limits<float>().max();
        ((float*)mGemmRelu.get())[1] = std::numeric_limits<float>().max();
        if (mBytes == 2) {
            gcore->MNNFp32ToLowp((float*)mGemmRelu.get(), reinterpret_cast<int16_t*>(mGemmRelu.get()), 2);
        }

        // GemmInt8 kernels
        if (mBytes == 4) {
            mInt8GemmKernel = core->Int8GemmKernel;
        } else {
            mInt8GemmKernel = core->MNNGemmInt8AddBiasScale_Unit_FP16;
        }

        if (mQuantValue) {
            mQuantQK = bufferAlloc->alloc(mThreadNum * eP8 * ROUND_UP(MNN_FLASH_ATTENTION_BLOCK_SIZE, mPack));
            mQKScale = bufferAlloc->alloc(eP8 * QUANT_INFO_BYTES);
            mQKBias = bufferAlloc->alloc(eP8 * QUANT_INFO_BYTES);
            mSumQK = bufferAlloc->alloc(mThreadNum * eP8 * QUANT_INFO_BYTES);

            if (mQuantQK.invalid() || mQKScale.invalid() || mQKBias.invalid() || mSumQK.invalid()) {
                return OUT_OF_MEMORY;
            }
        }
    } else {
        mPackQ.reset(Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(seqLen, eP), ROUND_UP(mHeadDim, lP), eP * mBytes}));
        backend()->onAcquireBuffer(mPackQ.get(), Backend::DYNAMIC);
        backend()->onAcquireBuffer(mPackQKV.get(), Backend::DYNAMIC);
    }

    // release tensor
    backend()->onReleaseBuffer(mPackQ.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mPackQKV.get(), Backend::DYNAMIC);

    if (inputs.size() > 4 || mUseFlashAttention) {
        backend()->onReleaseBuffer(mRunningMax.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mRunningSum.get(), Backend::DYNAMIC);
    }
    if (mUseFlashAttention) {
        backend()->onReleaseBuffer(mExpfDiffMax.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mTempOut.get(), Backend::DYNAMIC);
    }

    // release memchunk
    if (mQuantKey) {
        bufferAlloc->free(mSumQ);
        bufferAlloc->free(mQueryScale);
        bufferAlloc->free(mQueryZeroPoint);
        bufferAlloc->free(mQueryQuantScale);
        bufferAlloc->free(mQueryQuantZero);
        bufferAlloc->free(mQuantQuery);
        if (mBlockNum > 1) {
            bufferAlloc->free(mAccumBuffer);
        }
        if (mQuantValue) {
            bufferAlloc->free(mQuantQK);
            bufferAlloc->free(mQKScale);
            bufferAlloc->free(mQKBias);
            bufferAlloc->free(mSumQK);
        }
    }

    // Only allocated for quantized Q&K
    if (mQuantKey) {
        if (mBytes == 4) {
            mQuantFunc = core->MNNFloat2Int8;
        } else {
            mQuantFunc = core->DynamicQuanInput_ARM82;
        }

    }
    return NO_ERROR;
}

ErrorCode CPUAttention::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto gcore  = static_cast<CPUBackend *>(backend())->functions();
    auto core   = static_cast<CPUBackend*>(backend())->int8Functions();
    auto query = inputs[0];
    auto key   = inputs[1];
    auto value = inputs[2];
    int seqLen = query->length(1);
    const Tensor* mask = nullptr;
    if (inputs.size() > 3) {
        mask = inputs[3];
    }
    const Tensor* sinks = nullptr;
    if (inputs.size() > 4) {
        sinks = inputs[4];
        MNN_ASSERT(sinks != nullptr);
        MNN_ASSERT(sinks->elementSize() == mNumHead)
    }
    int numHeadDiv = UP_DIV(mNumHead, mThreadNum);
    int group_size = mNumHead / mKvNumHead;
    // reduce the value of 'query' to avoid fp16 overflow
    float mScale = 1.0 / sqrt(mHeadDim);
    float q_scale = 1.0;
    if (mBytes == 2 && !mQuantKey) {
        // reduce the value of 'query' to 'query * FP16_QSCALE', avoid fp16 overflow
        FLOAT16_T minValue;
        FLOAT16_T maxValue;
        gcore->MNNCountMaxMinValue(query->host<float>(), (float*)(&minValue), (float*)(&maxValue), query->elementSize());
        float maxV = maxValue;
        float minV = minValue;
        float absMax = ALIMAX(fabsf(maxV), fabsf(minV));
        if (absMax > 1.0f) {
            q_scale = 1.0f / absMax;
        }
        mScale /= q_scale;
    }
    int insertLen = seqLen;
    const int layerCount = (mMeta != nullptr) ? ALIMAX(1, mMeta->layer_nums) : 1;
    if (mH2OState != nullptr && (int)mH2OState->layerStates.size() < layerCount) {
        // Keep existing per-layer stats when a transient op reports smaller layer_count.
        // Only grow capacity; never shrink during an active run.
        mH2OState->layerStates.resize(layerCount);
        mH2OState->layerCacheManagers.resize(layerCount, nullptr);
    }
    const int preLayerCount = layerCount;
    int preLayerIndex = 0;
    if (mMeta != nullptr && mMeta->h2o_in_decode != 0 && mH2OState != nullptr) {
        preLayerIndex = static_cast<int>(mH2OState->decodeLayerCursor % static_cast<int64_t>(preLayerCount));
    }
    const bool runtimeStoreMode = mMeta != nullptr
        && mMeta->h2o_lossless_enable != 0
        && mMeta->h2o_lossless_runtime_enable != 0
        && mMeta->h2o_lossless_runtime_mode == 2;
    if (runtimeStoreMode
        && mKVCache
        && mH2OState != nullptr
        && preLayerIndex >= 0
        && preLayerIndex < (int)mH2OState->layerStates.size()) {
        struct RestoreChunk {
            int startToken = 0;
            int tokenCount = 0;
            uint64_t decodedBytes = 0;
            int64_t restoreUs = 0;
            std::vector<uint8_t> keyDecoded;
            std::vector<uint8_t> valueDecoded;
        };
        auto& restoreState = mH2OState->layerStates[preLayerIndex];
        const int flashBlockKv = ALIMAX(1, (int)mKVCacheManager->getFlashAttentionBlockKv());
        const bool strictRoundtrip = (mMeta->h2o_lossless_strict_roundtrip_check != 0);
        const int decodeCacheBlocks = ALIMAX(0, mMeta->h2o_lossless_decode_cache_blocks);
        const int globalDecodeCacheBlocks = decodeCacheBlocks;
        if (decodeCacheBlocks <= 0) {
            restoreState.decodeCacheEntries.clear();
            mH2OState->globalDecodeCacheEntries.clear();
        }
        auto appendLocalDecodeCache = [&](int startToken,
                                          int tokenCount,
                                          uint64_t rawHash,
                                          uint64_t blobHash,
                                          uint64_t rawBytes,
                                          uint64_t compressedBytes,
                                          const std::vector<uint8_t>& keyDecoded,
                                          const std::vector<uint8_t>& valueDecoded) {
            if (decodeCacheBlocks <= 0) {
                return;
            }
            for (auto cIt = restoreState.decodeCacheEntries.begin();
                 cIt != restoreState.decodeCacheEntries.end();) {
                const bool posMatch = cIt->startToken == startToken && cIt->tokenCount == tokenCount;
                const bool bytesMatch = cIt->rawBytes == rawBytes && cIt->compressedBytes == compressedBytes;
                const bool hashMatch = (rawHash != 0 && cIt->rawHash == rawHash)
                    || (blobHash != 0 && cIt->blobHash == blobHash);
                if ((posMatch && bytesMatch) || (bytesMatch && hashMatch)) {
                    cIt = restoreState.decodeCacheEntries.erase(cIt);
                } else {
                    ++cIt;
                }
            }
            CPUAttention::H2OSharedState::LayerState::DecodedCacheEntry entry;
            entry.startToken = startToken;
            entry.tokenCount = tokenCount;
            entry.rawHash = rawHash;
            entry.blobHash = blobHash;
            entry.rawBytes = rawBytes;
            entry.compressedBytes = compressedBytes;
            entry.keyDecoded = keyDecoded;
            entry.valueDecoded = valueDecoded;
            restoreState.decodeCacheEntries.push_back(std::move(entry));
            while ((int)restoreState.decodeCacheEntries.size() > decodeCacheBlocks) {
                restoreState.decodeCacheEntries.pop_front();
            }
        };
        auto appendGlobalDecodeCache = [&](int startToken,
                                           int tokenCount,
                                           uint64_t rawHash,
                                           uint64_t blobHash,
                                           uint64_t rawBytes,
                                           uint64_t compressedBytes,
                                           const std::vector<uint8_t>& keyDecoded,
                                           const std::vector<uint8_t>& valueDecoded) {
            if (globalDecodeCacheBlocks <= 0) {
                return;
            }
            auto& globalCache = mH2OState->globalDecodeCacheEntries;
            for (auto cIt = globalCache.begin(); cIt != globalCache.end();) {
                const bool bytesMatch = cIt->rawBytes == rawBytes && cIt->compressedBytes == compressedBytes;
                const bool hashMatch = (rawHash != 0 && cIt->rawHash == rawHash)
                    || (blobHash != 0 && cIt->blobHash == blobHash);
                if (bytesMatch && hashMatch) {
                    cIt = globalCache.erase(cIt);
                } else {
                    ++cIt;
                }
            }
            CPUAttention::H2OSharedState::LayerState::DecodedCacheEntry entry;
            entry.startToken = startToken;
            entry.tokenCount = tokenCount;
            entry.rawHash = rawHash;
            entry.blobHash = blobHash;
            entry.rawBytes = rawBytes;
            entry.compressedBytes = compressedBytes;
            entry.keyDecoded = keyDecoded;
            entry.valueDecoded = valueDecoded;
            globalCache.push_back(std::move(entry));
            while ((int)globalCache.size() > globalDecodeCacheBlocks) {
                globalCache.pop_front();
            }
        };
        std::vector<RestoreChunk> restoreChunks;
        restoreChunks.reserve(restoreState.losslessBlocks.size());
        for (auto it = restoreState.losslessBlocks.begin(); it != restoreState.losslessBlocks.end();) {
            auto& block = *it;
            if (!block.rawDropped) {
                // Store mode only needs compressed payload until first successful restore.
                // Drop stale restored blocks to cap long-run memory usage.
                it = restoreState.losslessBlocks.erase(it);
                continue;
            }

            const uint64_t blobHash = (block.blobHash != 0)
                ? block.blobHash
                : fnv1a64(block.valueBlob.data(), block.valueBlob.size(),
                          fnv1a64(block.keyBlob.data(), block.keyBlob.size()));
            std::vector<uint8_t> keyDecoded;
            std::vector<uint8_t> valueDecoded;
            bool cacheHit = false;
            auto tryDecodeCache = [&](bool requirePosShape) {
                for (auto cIt = restoreState.decodeCacheEntries.begin(); cIt != restoreState.decodeCacheEntries.end(); ++cIt) {
                    if (requirePosShape) {
                        const bool shapeMatch = cIt->startToken == block.startToken
                            && cIt->tokenCount == block.tokenCount;
                        if (!shapeMatch) {
                            continue;
                        }
                    }
                    const bool bytesMatch = cIt->rawBytes == block.rawBytes
                        && cIt->compressedBytes == block.compressedBytes;
                    if (!bytesMatch) {
                        continue;
                    }
                    const bool hashMatch = (block.rawHash != 0 && cIt->rawHash == block.rawHash)
                        || (blobHash != 0 && cIt->blobHash == blobHash);
                    if (!hashMatch) {
                        continue;
                    }
                    keyDecoded = cIt->keyDecoded;
                    valueDecoded = cIt->valueDecoded;
                    auto entry = std::move(*cIt);
                    restoreState.decodeCacheEntries.erase(cIt);
                    restoreState.decodeCacheEntries.push_back(std::move(entry));
                    cacheHit = true;
                    mH2OState->globalLosslessDecodeCacheHit += 1;
                    return;
                }
            };
            auto restoreT0 = std::chrono::high_resolution_clock::now();
            tryDecodeCache(true);
            if (!cacheHit) {
                // Store-mode eviction can renumber token positions; allow
                // hash+bytes match fallback for eviction-invariant reuse.
                tryDecodeCache(false);
            }
            if (!cacheHit && globalDecodeCacheBlocks > 0) {
                auto& globalCache = mH2OState->globalDecodeCacheEntries;
                for (auto cIt = globalCache.begin(); cIt != globalCache.end(); ++cIt) {
                    const bool bytesMatch = cIt->rawBytes == block.rawBytes
                        && cIt->compressedBytes == block.compressedBytes;
                    if (!bytesMatch) {
                        continue;
                    }
                    const bool hashMatch = (block.rawHash != 0 && cIt->rawHash == block.rawHash)
                        || (blobHash != 0 && cIt->blobHash == blobHash);
                    if (!hashMatch) {
                        continue;
                    }
                    keyDecoded = cIt->keyDecoded;
                    valueDecoded = cIt->valueDecoded;
                    auto entry = std::move(*cIt);
                    globalCache.erase(cIt);
                    globalCache.push_back(std::move(entry));
                    cacheHit = true;
                    mH2OState->globalLosslessDecodeCacheHit += 1;
                    appendLocalDecodeCache(
                        block.startToken,
                        block.tokenCount,
                        block.rawHash,
                        blobHash,
                        block.rawBytes,
                        block.compressedBytes,
                        keyDecoded,
                        valueDecoded);
                    break;
                }
            }

            bool fallbackUsed = false;
            if (!cacheHit) {
                bool keyOk = true;
                bool valueOk = true;
                if (!block.keyBlob.empty()) {
                    if (mBytes == 2) {
                        keyOk = decodeFp16GearPredictive(block.keyBlob.data(), block.keyBlob.size(), keyDecoded);
                    } else if (mBytes == 4) {
                        keyOk = decodeFp32LanePredictive(block.keyBlob.data(), block.keyBlob.size(), keyDecoded);
                    } else {
                        keyOk = false;
                    }
                }
                if (!block.valueBlob.empty()) {
                    if (mBytes == 2) {
                        valueOk = decodeFp16GearPredictive(block.valueBlob.data(), block.valueBlob.size(), valueDecoded);
                    } else if (mBytes == 4) {
                        valueOk = decodeFp32LanePredictive(block.valueBlob.data(), block.valueBlob.size(), valueDecoded);
                    } else {
                        valueOk = false;
                    }
                }
                mH2OState->globalLosslessDecodeCacheMiss += 1;
                if (!keyOk || !valueOk) {
                    fallbackUsed = true;
                }
            }
            auto restoreT1 = std::chrono::high_resolution_clock::now();
            int64_t restoreUs = (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(restoreT1 - restoreT0).count();
            if (restoreUs <= 0 && (cacheHit || !keyDecoded.empty() || !valueDecoded.empty())) {
                restoreUs = 1;
            }
            const uint64_t decodedBytes = (uint64_t)keyDecoded.size() + (uint64_t)valueDecoded.size();
            if (!fallbackUsed && decodedBytes != block.rawBytes) {
                fallbackUsed = true;
            }
            if (!fallbackUsed && strictRoundtrip) {
                if (block.rawHash == 0) {
                    fallbackUsed = true;
                } else {
                    uint64_t decodedHash = fnv1a64(keyDecoded.data(), keyDecoded.size());
                    decodedHash = fnv1a64(valueDecoded.data(), valueDecoded.size(), decodedHash);
                    if (decodedHash != block.rawHash) {
                        fallbackUsed = true;
                    }
                }
            }
            if (!fallbackUsed) {
                if (!cacheHit) {
                    appendLocalDecodeCache(
                        block.startToken,
                        block.tokenCount,
                        block.rawHash,
                        blobHash,
                        block.rawBytes,
                        block.compressedBytes,
                        keyDecoded,
                        valueDecoded);
                }
                appendGlobalDecodeCache(
                    block.startToken,
                    block.tokenCount,
                    block.rawHash,
                    blobHash,
                    block.rawBytes,
                    block.compressedBytes,
                    keyDecoded,
                    valueDecoded);
            }

            if (!fallbackUsed) {
                RestoreChunk chunk;
                chunk.startToken = block.startToken;
                chunk.tokenCount = block.tokenCount;
                chunk.decodedBytes = decodedBytes;
                chunk.restoreUs = restoreUs;
                chunk.keyDecoded.swap(keyDecoded);
                chunk.valueDecoded.swap(valueDecoded);
                restoreChunks.emplace_back(std::move(chunk));
            } else {
                restoreState.losslessDecompressUs += restoreUs;
                restoreState.losslessFallbackCount += 1;
            }
            // Once restored (or failed irrecoverably), payload is no longer useful.
            it = restoreState.losslessBlocks.erase(it);
        }
        if (!restoreChunks.empty()) {
            std::sort(restoreChunks.begin(), restoreChunks.end(), [](const RestoreChunk& a, const RestoreChunk& b) {
                return a.startToken < b.startToken;
            });
            std::vector<RestoreChunk> mergedChunks;
            mergedChunks.reserve(restoreChunks.size());
            for (auto& chunk : restoreChunks) {
                if (!mergedChunks.empty()) {
                    auto& last = mergedChunks.back();
                    const int lastEnd = last.startToken + last.tokenCount;
                    if (lastEnd == chunk.startToken) {
                        last.tokenCount += chunk.tokenCount;
                        last.decodedBytes += chunk.decodedBytes;
                        last.restoreUs += chunk.restoreUs;
                        last.keyDecoded.insert(last.keyDecoded.end(), chunk.keyDecoded.begin(), chunk.keyDecoded.end());
                        last.valueDecoded.insert(last.valueDecoded.end(), chunk.valueDecoded.begin(), chunk.valueDecoded.end());
                        continue;
                    }
                }
                mergedChunks.emplace_back(std::move(chunk));
            }
            for (auto& chunk : mergedChunks) {
                auto scatterT0 = std::chrono::high_resolution_clock::now();
                const bool restored = scatterKvMergedRange(
                    mKVCacheManager.get(),
                    mKvNumHead,
                    mHeadDim,
                    mBytes,
                    lP,
                    hP,
                    flashBlockKv,
                    chunk.startToken,
                    chunk.tokenCount,
                    chunk.keyDecoded,
                    chunk.valueDecoded);
                auto scatterT1 = std::chrono::high_resolution_clock::now();
                int64_t scatterUs = (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(scatterT1 - scatterT0).count();
                if (scatterUs <= 0 && chunk.decodedBytes > 0) {
                    scatterUs = 1;
                }
                restoreState.losslessDecompressUs += chunk.restoreUs + scatterUs;
                if (restored) {
                    restoreState.losslessDecompressedBytes += chunk.decodedBytes;
                } else {
                    restoreState.losslessFallbackCount += 1;
                    MNN_ERROR("[H2O-LOSSLESS] scatter restore failed: layer=%d start=%d tokens=%d active_kv=%d max_kv=%d\n",
                              preLayerIndex,
                              chunk.startToken,
                              chunk.tokenCount,
                              mKVCacheManager != nullptr ? mKVCacheManager->kvLength() : -1,
                              mKVCacheManager != nullptr ? mKVCacheManager->maxLength() : -1);
                }
            }
        }
    }

    if (mKVCache && mMeta != nullptr) {
        const bool h2oDecodeStep = (mMeta->h2o_enable != 0) && (mMeta->h2o_in_decode != 0);
        if (h2oDecodeStep) {
            // `remove/reserve` in KVMeta are shared across all layers in one forward.
            // H2O pending compact plans are now per-layer, so clear any carry-over
            // from the previous layer before applying this layer's pending plan.
            mMeta->remove = 0;
            mMeta->reserve = nullptr;
            mMeta->n_reserve = 0;
        }
        const size_t kvLenBeforeRealloc = (size_t)ALIMAX(0, mKVCacheManager->kvLength());
        // Apply previous-step H2O compact plan from per-layer storage.
        // Using layer-local buffers avoids cross-layer raw-pointer aliasing.
        if (mH2OState != nullptr
            && preLayerIndex >= 0
            && preLayerIndex < (int)mH2OState->layerStates.size()) {
            auto& pendingState = mH2OState->layerStates[preLayerIndex];
            if (pendingState.pendingPlanReady != 0) {
                const size_t pendingRemove = pendingState.pendingRemove;
                const int pendingInts = (int)pendingState.pendingReservePairs.size();
                const int pendingPairs = pendingInts / 2;
                bool validPending = ((pendingInts % 2) == 0)
                    && (pendingRemove == kvLenBeforeRealloc);
                if (validPending && pendingPairs > 0) {
                    int64_t lastEnd = 0;
                    for (int n = 0; n < pendingPairs; ++n) {
                        const int begin = pendingState.pendingReservePairs[2 * n];
                        const int size = pendingState.pendingReservePairs[2 * n + 1];
                        const int64_t end = (int64_t)begin + (int64_t)size;
                        if (begin < 0
                            || size <= 0
                            || end < (int64_t)begin
                            || end > (int64_t)pendingRemove
                            || (int64_t)begin < lastEnd) {
                            validPending = false;
                            break;
                        }
                        lastEnd = end;
                    }
                }
                if (validPending) {
                    pendingState.activeReservePairs.swap(pendingState.pendingReservePairs);
                    mMeta->remove = pendingRemove;
                    mMeta->reserve = pendingState.activeReservePairs.empty()
                        ? nullptr
                        : pendingState.activeReservePairs.data();
                    mMeta->n_reserve = (int)pendingState.activeReservePairs.size() / 2;
                } else if (!validPending) {
                    MNN_ERROR("Drop stale per-layer pending plan: layer=%d pending_remove=%zu kv_len=%zu pending_ints=%d.\n",
                              preLayerIndex,
                              pendingRemove,
                              kvLenBeforeRealloc,
                              pendingInts);
                }
                pendingState.pendingPlanReady = 0;
                pendingState.pendingRemove = 0;
                pendingState.pendingReservePairs.clear();
            }
        }
        // Do not carry raw pending pointers in shared meta across layers.
        mMeta->h2o_pending_plan_ready = 0;
        mMeta->h2o_pending_remove = 0;
        mMeta->h2o_pending_reserve = nullptr;
        mMeta->h2o_pending_n_reserve = 0;
        if (mMeta->remove > kvLenBeforeRealloc) {
            MNN_ERROR("Clamp invalid meta remove: remove=%zu > kv_len=%zu.\n",
                      mMeta->remove,
                      kvLenBeforeRealloc);
            mMeta->remove = kvLenBeforeRealloc;
            mMeta->reserve = nullptr;
            mMeta->n_reserve = 0;
        }
        if (mMeta->n_reserve < 0) {
            MNN_ERROR("Clamp invalid meta n_reserve=%d to 0.\n", mMeta->n_reserve);
            mMeta->n_reserve = 0;
            mMeta->reserve = nullptr;
        }
        if (mMeta->n_reserve > 0) {
            bool validReserve = (mMeta->reserve != nullptr) && (mMeta->remove > 0);
            int64_t lastEnd = 0;
            if (validReserve) {
                for (int n = 0; n < mMeta->n_reserve; ++n) {
                    const int begin = mMeta->reserve[2 * n];
                    const int size = mMeta->reserve[2 * n + 1];
                    const int64_t end = static_cast<int64_t>(begin) + static_cast<int64_t>(size);
                    if (begin < 0
                        || size <= 0
                        || end < static_cast<int64_t>(begin)
                        || end > static_cast<int64_t>(mMeta->remove)
                        || static_cast<int64_t>(begin) < lastEnd) {
                        validReserve = false;
                        break;
                    }
                    lastEnd = end;
                }
            }
            if (!validReserve) {
                MNN_ERROR("Drop invalid meta reserve before realloc: remove=%zu n_reserve=%d.\n",
                          mMeta->remove,
                          mMeta->n_reserve);
                mMeta->reserve = nullptr;
                mMeta->n_reserve = 0;
            }
        }
        if (mMeta->previous == mMeta->remove && mMeta->n_reserve == 0) {
            mKVCacheManager->onClear();
            mKVCacheManager->onAlloc(mMeta, seqLen);
        } else {
            mKVCacheManager->onRealloc(mMeta);
        }
        insertLen = (int)mMeta->add;
    } else {
        mKVCacheManager->onClear();
        mKVCacheManager->onAlloc(mMeta, seqLen);
    }

    // Add the new kv to the kvcache
    mKVCacheManager->onUpdateKV(key, value, (int)insertLen);

    if (mUseFlashAttention) {
        mBlockKV = ALIMIN(MNN_FLASH_ATTENTION_BLOCK_SIZE, mKVCacheManager->kvLength());
    } else {
        mBlockKV = mKVCacheManager->kvLength();
    }

    // Constant Initialization
    auto padSeqLength = seqLen - insertLen;
    seqLen = insertLen;
    int kvSeqLen  = mKVCacheManager->kvLength();
    int maxLen = mKVCacheManager->maxLength();
    int32_t units[2] = {eP, lP};
    const float* sinksPtr = sinks ? sinks->host<float>() : nullptr;
    int kvValidOffset = kvSeqLen - seqLen; // reuse_kv=true or decode, kvValidOffset>0

    auto teardownAsyncWorkers = [&](bool clearCompleted) {
        if (mH2OState == nullptr) {
            return;
        }
        std::vector<std::thread> workers;
        {
            std::lock_guard<std::mutex> lock(mH2OState->asyncMutex);
            mH2OState->asyncStop = true;
            mH2OState->asyncCv.notify_all();
            workers.swap(mH2OState->asyncWorkers);
        }
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        {
            std::lock_guard<std::mutex> lock(mH2OState->asyncMutex);
            mH2OState->asyncStop = false;
            mH2OState->asyncConfiguredThreads = 0;
            mH2OState->asyncRunningTasks = 0;
            mH2OState->asyncPendingTasks = 0;
            mH2OState->asyncTasks.clear();
            if (clearCompleted) {
                mH2OState->asyncCompleted.clear();
            }
        }
    };
    auto ensureAsyncWorkers = [&](int desiredThreads) {
        if (mH2OState == nullptr) {
            return;
        }
        desiredThreads = ALIMAX(0, desiredThreads);
        int currentThreads = 0;
        {
            std::lock_guard<std::mutex> lock(mH2OState->asyncMutex);
            currentThreads = mH2OState->asyncConfiguredThreads;
            if (currentThreads == desiredThreads && (int)mH2OState->asyncWorkers.size() == desiredThreads) {
                return;
            }
        }

        teardownAsyncWorkers(true);
        if (desiredThreads <= 0) {
            return;
        }

        auto sharedState = mH2OState;
        std::vector<std::thread> workers;
        workers.reserve(desiredThreads);
        for (int i = 0; i < desiredThreads; ++i) {
            workers.emplace_back([sharedState]() {
                while (true) {
                    CPUAttention::H2OSharedState::AsyncTask task;
                    {
                        std::unique_lock<std::mutex> lock(sharedState->asyncMutex);
                        sharedState->asyncCv.wait(lock, [&]() {
                            return sharedState->asyncStop || !sharedState->asyncTasks.empty();
                        });
                        if (sharedState->asyncStop && sharedState->asyncTasks.empty()) {
                            return;
                        }
                        task = std::move(sharedState->asyncTasks.front());
                        sharedState->asyncTasks.pop_front();
                        sharedState->asyncRunningTasks += 1;
                    }

                    CPUAttention::H2OSharedState::AsyncResult result;
                    result.taskId = task.taskId;
                    result.layerIndex = task.layerIndex;
                    result.startToken = task.startToken;
                    result.tokenCount = task.tokenCount;
                    result.runtimeStoreMode = task.runtimeStoreMode;
                    if (task.fn) {
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
                        try {
                            result = task.fn();
                            result.taskId = task.taskId;
                        } catch (...) {
                            result.fallbackUsed = true;
                        }
#else
                        result = task.fn();
                        result.taskId = task.taskId;
#endif
                    } else {
                        result.fallbackUsed = true;
                    }

                    {
                        std::lock_guard<std::mutex> lock(sharedState->asyncMutex);
                        sharedState->asyncRunningTasks -= 1;
                        sharedState->asyncPendingTasks -= 1;
                        if (sharedState->asyncRunningTasks < 0 || sharedState->asyncPendingTasks < 0) {
                            MNN_ERROR("[H2O-LOSSLESS] async counters out-of-sync: running=%lld pending=%lld\n",
                                      (long long)sharedState->asyncRunningTasks,
                                      (long long)sharedState->asyncPendingTasks);
                            sharedState->asyncRunningTasks = std::max<int64_t>(0, sharedState->asyncRunningTasks);
                            sharedState->asyncPendingTasks = std::max<int64_t>(0, sharedState->asyncPendingTasks);
                        }
                        sharedState->asyncCompleted.emplace_back(std::move(result));
                    }
                    sharedState->asyncDoneCv.notify_all();
                }
            });
        }
        {
            std::lock_guard<std::mutex> lock(mH2OState->asyncMutex);
            mH2OState->asyncConfiguredThreads = desiredThreads;
            for (auto& worker : workers) {
                mH2OState->asyncWorkers.emplace_back(std::move(worker));
            }
        }
    };
    const bool runtimeLosslessEnabledGlobal = mMeta != nullptr
        && mMeta->h2o_lossless_enable != 0
        && mMeta->h2o_lossless_runtime_enable != 0;
    const bool runtimeProbeModeGlobal = mMeta != nullptr && mMeta->h2o_lossless_runtime_mode == 0;
    const int desiredAsyncThreads = (runtimeLosslessEnabledGlobal && !runtimeProbeModeGlobal)
        ? ALIMAX(0, mMeta->h2o_lossless_async_threads)
        : 0;
    ensureAsyncWorkers(desiredAsyncThreads);

    int layerIndex = 0;
    if (mMeta != nullptr && mMeta->h2o_in_decode != 0 && mH2OState != nullptr) {
        layerIndex = static_cast<int>(mH2OState->decodeLayerCursor % static_cast<int64_t>(layerCount));
        mH2OState->decodeLayerCursor += 1;
        mH2OState->globalStep += 1;
        if (layerIndex == 0) {
            // Token-level clock for expensive lossless runtime updates.
            mH2OState->globalTokenStep += 1;
        }
    } else if (mMeta != nullptr && mH2OState != nullptr && kvSeqLen <= seqLen) {
        // New prefill/request path: clear decode-side lossless running stats.
        // Guard by kvSeqLen<=seqLen to avoid accidental reset during decode if h2o_in_decode flag jitters.
        mH2OState->decodeLayerCursor = 0;
        mH2OState->globalStep = 0;
        mH2OState->globalTokenStep = 0;
        mH2OState->globalLastTriggerStep = 0;
        mH2OState->globalLastLosslessStep = 0;
        mH2OState->globalLastLosslessTokenBudget = 0;
        mH2OState->globalLastLosslessRatio = 1.0f;
        mH2OState->globalLastLosslessCodecUs = 0;
        mH2OState->globalLastLosslessRawBytes = 0;
        mH2OState->globalLastLosslessCompressedBytes = 0;
        mH2OState->globalLastLosslessDecompressedBytes = 0;
        mH2OState->globalLastLosslessCompressUs = 0;
        mH2OState->globalLastLosslessDecompressUs = 0;
        mH2OState->globalLosslessQueueDepthPeak = 0;
        mH2OState->globalLosslessFallbackCount = 0;
        mH2OState->globalLosslessBackpressureSkipCount = 0;
        mH2OState->globalLosslessAsyncQueuePeak = 0;
        mH2OState->globalLosslessAsyncWaitUs = 0;
        mH2OState->globalLosslessDecodeCacheHit = 0;
        mH2OState->globalLosslessDecodeCacheMiss = 0;
        mH2OState->reserveStorage.clear();
        mH2OState->reserveStoragePending.clear();
        {
            std::unique_lock<std::mutex> lock(mH2OState->asyncMutex);
            if (mH2OState->asyncPendingTasks > 0) {
                auto w0 = std::chrono::high_resolution_clock::now();
                mH2OState->asyncDoneCv.wait(lock, [&]() {
                    return mH2OState->asyncPendingTasks == 0;
                });
                auto w1 = std::chrono::high_resolution_clock::now();
                mH2OState->globalLosslessAsyncWaitUs +=
                    (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(w1 - w0).count();
            }
            mH2OState->asyncTasks.clear();
            mH2OState->asyncCompleted.clear();
            mH2OState->asyncPendingTasks = 0;
            mH2OState->asyncRunningTasks = 0;
        }
        for (auto& state : mH2OState->layerStates) {
            state.blockScores.clear();
            state.pendingRemove = 0;
            state.pendingPlanReady = 0;
            state.activeReservePairs.clear();
            state.pendingReservePairs.clear();
            state.losslessLastStep = 0;
            state.losslessLastTokenBudget = 0;
            state.losslessRawBytes = 0;
            state.losslessCompressedBytes = 0;
            state.losslessDecompressedBytes = 0;
            state.losslessCompressUs = 0;
            state.losslessDecompressUs = 0;
            state.losslessFallbackCount = 0;
            state.losslessUpdateCount = 0;
            state.losslessBackpressureSkipCount = 0;
            state.losslessBlocks.clear();
            state.decodeCacheEntries.clear();
        }
        mMeta->remove = 0;
        mMeta->reserve = nullptr;
        mMeta->n_reserve = 0;
        mMeta->h2o_pending_plan_ready = 0;
        mMeta->h2o_pending_remove = 0;
        mMeta->h2o_pending_reserve = nullptr;
        mMeta->h2o_pending_n_reserve = 0;
    }
    if (mH2OState != nullptr) {
        if ((int)mH2OState->layerCacheManagers.size() < layerCount) {
            mH2OState->layerCacheManagers.resize(layerCount, nullptr);
        }
        if (layerIndex >= 0 && layerIndex < (int)mH2OState->layerCacheManagers.size()) {
            mH2OState->layerCacheManagers[layerIndex] = mKVCacheManager.get();
        }
    }
    int h2oLayerStart = 0;
    int h2oLayerEnd = layerCount - 1;
    if (mMeta != nullptr) {
        h2oLayerStart = clampInt(mMeta->h2o_layer_start, 0, layerCount - 1);
        if (mMeta->h2o_layer_end >= 0) {
            h2oLayerEnd = clampInt(mMeta->h2o_layer_end, 0, layerCount - 1);
        }
    }
    const bool h2oLayerInRange = (h2oLayerEnd >= h2oLayerStart)
        && (layerIndex >= h2oLayerStart)
        && (layerIndex <= h2oLayerEnd);

    const bool h2oEnabled = mKVCache
        && mMeta != nullptr
        && mMeta->h2o_enable != 0
        && mMeta->h2o_in_decode != 0
        && h2oLayerInRange
        && !mQuantKey
        && !mQuantValue
        && seqLen > 0
        && kvSeqLen > 0;
    const int h2oBlockTokens = h2oEnabled ? ALIMAX(1, mMeta->h2o_block_tokens) : 1;
    const int h2oBlockCount = h2oEnabled ? UP_DIV(kvSeqLen, h2oBlockTokens) : 0;
    std::vector<float> h2oBlockAcc;
    std::mutex h2oBlockLock;
    if (h2oEnabled && h2oBlockCount > 0) {
        h2oBlockAcc.resize(h2oBlockCount, 0.0f);
    }

    // Temporary tensors for intermediate results
    std::shared_ptr<Tensor> unpackQK(Tensor::createDevice<int32_t>({mThreadNum, seqLen, mBlockKV}));
    std::shared_ptr<Tensor> softmMaxQ(Tensor::createDevice<int32_t>({mThreadNum, seqLen, ROUND_UP(mBlockKV, mPack)})); // [mBlockKV/mPack, seqLen, mPack ]
    std::shared_ptr<Tensor> newPackQK;
    if (mQuantValue == false) {
        newPackQK.reset(Tensor::createDevice<int8_t>({mThreadNum, eP * ROUND_UP(mBlockKV, lP) * mBytes}));
    } else {
        newPackQK.reset(Tensor::createDevice<int8_t>({mThreadNum, eP8 * ROUND_UP(MNN_FLASH_ATTENTION_BLOCK_SIZE, lP8)}));
    }
    std::shared_ptr<Tensor> mTempQKBlock(Tensor::createDevice<int8_t>({mThreadNum, UP_DIV(mBlockKV, mPack), seqLen, mPack * mBytes}));
    backend()->onAcquireBuffer(unpackQK.get(), Backend::STATIC);
    backend()->onAcquireBuffer(softmMaxQ.get(), Backend::STATIC);
    backend()->onAcquireBuffer(newPackQK.get(), Backend::STATIC);
    backend()->onAcquireBuffer(mTempQKBlock.get(), Backend::STATIC);

    // Quantize Q and initialize bias 0
    if (mQuantKey) {
        mGemmBias.reset(ROUND_UP(ALIMAX(mBlockKV, mHeadDim), hP8) * QUANT_INFO_BYTES);
        if (!mGemmBias.get()) {
            MNN_ERROR("Allocate bias buffer failed in CPU Attention\n");
            return OUT_OF_MEMORY;
        }
        memset(mGemmBias.get(), 0, ROUND_UP(ALIMAX(mBlockKV, mHeadDim), hP8) * QUANT_INFO_BYTES);

        // Q: [seqLen,numHead,headDim]
        // maxQ, minQ: [seqLen,numHead]
        // scaleQ, zeroQ: [numHead, seqLen]
        // quantQ: [seqLen,numHead,headDim]
        auto queryPtr = query->host<int8_t>();
        int divPart = UP_DIV(seqLen * mNumHead, mThreadNum);
        MNN_CONCURRENCY_BEGIN (tId, mThreadNum) {
            size_t info[9] = {1, (size_t)mHeadDim, 1, 1, 1, 1, 1, 1, 0};
            auto remainLu = seqLen * mNumHead - tId * divPart;
            if (remainLu > 0) {
                remainLu = ALIMIN(divPart, remainLu);
                for (int i = tId * divPart; i < tId * divPart + remainLu; ++i) {

                    // address
                    auto srcFloatPtr = (float*)(queryPtr + i * mHeadDim * mBytes);
                    auto dstInt8Ptr = (int8_t*)(mQuantQuery.ptr() + i * mHeadDim);
                    auto quantScalePtr = (float*)(mQueryQuantScale.ptr() + i * QUANT_INFO_BYTES);
                    auto quantZeroPtr = (float*)(mQueryQuantZero.ptr() + i * QUANT_INFO_BYTES);

                    // scaleQ, zeroQ, [seqLen,numHead]->[numHead,seqLen]
                    int indexQ = (i / mNumHead) + (i % mNumHead) * seqLen;
                    auto scalePtr = (float*)(mQueryScale.ptr() + indexQ * QUANT_INFO_BYTES);
                    auto zeroPtr = (float*)(mQueryZeroPoint.ptr() + indexQ * QUANT_INFO_BYTES);


                    // compute the quant/dequant scale/bias
                    gcore->MNNAsyQuantInfo(scalePtr, zeroPtr, quantScalePtr, quantZeroPtr, nullptr, nullptr, srcFloatPtr, info);
                    scalePtr[0] *= mScale;
                    zeroPtr[0] *= mScale;

                    // quantize the float query to int8_t query
                    mQuantFunc(srcFloatPtr, dstInt8Ptr, UP_DIV(mHeadDim, gcore->pack), quantScalePtr, -128, 127, quantZeroPtr, 0);
                }
            }
        } MNN_CONCURRENCY_END();

        // source int8_t query: [seqLen,numHead,headDim]
        // dest int8_t query: [numHead,seqLen/eP,headDim/lP,eP,lP]

        int outterSeqLen = UP_DIV(seqLen, eP8);
        int outterHeadDim = UP_DIV(mHeadDim, lP8);
        size_t outputOffset = 0;

        const int8_t* src_base_ptr = (const int8_t*)mQuantQuery.ptr();
        int8_t* dst_base_ptr = mPackQ->host<int8_t>();

        for (int h = 0; h < mNumHead; ++h) {
            for (int seqBlock = 0; seqBlock < outterSeqLen; ++seqBlock) {
                int seqBase = seqBlock * eP8;
                int eunit = std::min(eP8, seqLen - seqBase);
                size_t currentSeqBlockSize = (size_t)outterHeadDim * eunit * lP8;

                for (int dimBlock = 0; dimBlock < outterHeadDim; ++dimBlock) {
                    int dimBase = dimBlock * lP8;
                    int headDimRemain = mHeadDim - dimBase;
                    int copyLen = std::min(lP8, headDimRemain);

                    if (copyLen <= 0) {
                        continue;
                    }

                    int8_t* dst_block_ptr = dst_base_ptr +
                                          outputOffset +
                                          (size_t)dimBlock * (eunit * lP8);

                    const size_t src_row_stride = (size_t)mNumHead * mHeadDim;

                    for (int seqLocal = 0; seqLocal < eunit; ++seqLocal) {
                        int innerSeq = seqBase + seqLocal;

                        const int8_t* src_row_ptr = src_base_ptr +
                                                    (size_t)innerSeq * src_row_stride +
                                                    (size_t)h * mHeadDim +
                                                    dimBase;

                        int8_t* dst_row_ptr = dst_block_ptr + seqLocal * lP8;

                        std::memcpy(dst_row_ptr, src_row_ptr, copyLen);
                    }
                    if (copyLen < lP8) {
                        for (int seqLocal = 0; seqLocal < eunit; ++seqLocal) {
                            int8_t* dst_pad_ptr = dst_block_ptr + seqLocal * lP8 + copyLen;
                            std::memset(dst_pad_ptr, 0, lP8 - copyLen);
                        }
                    }
                }
                outputOffset += currentSeqBlockSize;
            }
        } // Finish quantize Q

        if (mQuantValue) {
            auto scalePtr = (float*)(mQKScale.ptr());
            auto zeroPtr = (float*)(mQKBias.ptr());
            for (int k = 0; k < eP8; ++k) {
                scalePtr[k] = 1.f / 255.f;
#ifdef MNN_USE_SSE
                zeroPtr[k] =0;
#else
                zeroPtr[k] = 128.f / 255.f;
#endif
            }
        }

    }

    std::function<void(int)> mCompute = [=, &h2oBlockAcc, &h2oBlockLock](int tId) {
        int8_t* qReordered = nullptr;
        auto qkPacked     = mTempQKBlock->host<int8_t>() + tId * mTempQKBlock->stride(0);
        auto qkFlatten   = unpackQK->host<float>() + tId * unpackQK->stride(0);
        auto qkSoftmax  = softmMaxQ->host<float>() + tId * softmMaxQ->stride(0);
        auto qkReordered = newPackQK->host<int8_t>() + tId * newPackQK->stride(0);
        auto qkvPacked    = mPackQKV->host<int8_t>() + tId * mPackQKV->stride(0);
        int  headIndex  = tId * numHeadDiv;
        int  headsToCompute = ALIMIN(numHeadDiv, mNumHead - headIndex);

        // Flash Attention
        auto runningMax = mRunningMax ? (float*)(mRunningMax->host<int8_t>() + tId * mRunningMax->stride(0)) : nullptr;
        auto runningSum = mRunningSum ? (float*)(mRunningSum->host<int8_t>() + tId * mRunningSum->stride(0)) : nullptr;
        auto diffScale = mExpfDiffMax ? (float*)(mExpfDiffMax->host<int8_t>() + tId * mExpfDiffMax->stride(0)) : nullptr;
        auto outputPacked = mTempOut ? mTempOut->host<int8_t>() + tId * mTempOut->stride(0) : qkvPacked;
        std::vector<float> localBlockAcc;
        if (h2oEnabled && h2oBlockCount > 0) {
            localBlockAcc.resize(h2oBlockCount, 0.0f);
        }
        
        int  kvBlocks = UP_DIV(kvSeqLen, mBlockKV);

        bool isLowerTriangular = (mask == nullptr);
        if (mask != nullptr && mask->shape().empty()) {
            if (mBytes == 2) {
                auto maskPtr = mask->host<FLOAT16_T>();
                if (maskPtr[0] < 1e-6) {
                    isLowerTriangular = true;
                }
            } else {
                auto maskPtr = mask->host<float>();
                if (maskPtr[0] < 1e-6f) {
                    isLowerTriangular = true;
                }
            }
        }
        bool useMaskInSoftmax = (isLowerTriangular && sinksPtr == nullptr);

        QuanPostTreatParameters gemmParam4QxK, gemmParam4QKxV; // used by int8 gemm, allocated per thread.
        SumByAxisParams sumParams4QxK, sumParams4QKxV = {};
        float* qSumAddr = nullptr;
        float* qScale = nullptr;
        float* qBias = nullptr;
        float* accumbuff = nullptr;
        int32_t unitColBufferSize = 0;
        if (mQuantKey) {
            // parameters shared by all mBlockKV
            gemmParam4QxK.blockNum = mBlockNum;
            gemmParam4QxK.biasFloat = reinterpret_cast<float*>(mGemmBias.get());
            gemmParam4QxK.useInt8 = 0;
            gemmParam4QxK.fp32minmax = reinterpret_cast<float*>(mGemmRelu.get());

            sumParams4QxK.oneScale = 0;
            sumParams4QxK.SRC_UNIT = lP8;
            sumParams4QxK.blockNum = mBlockNum;
            sumParams4QxK.DST_XUNIT = eP8;
            sumParams4QxK.inputBlock = 0;
            sumParams4QxK.kernelxy = 1;
            // fixed
            sumParams4QxK.LU = UP_DIV(mHeadDim, lP8);
            sumParams4QxK.unitColBufferSize = ROUND_UP(mHeadDim, lP8) * eP8;
            sumParams4QxK.kernelCountUnitDouble = UP_DIV(mHeadDim, lP8);
            sumParams4QxK.valid = mHeadDim % lP8;


            if (mBlockNum > 1) {
                accumbuff = (float*)(mAccumBuffer.ptr() + tId * eP8 * hP8 * QUANT_INFO_BYTES);
            }
            unitColBufferSize = eP8 * ROUND_UP(mHeadDim, lP8);

            if (mQuantValue) {
                gemmParam4QKxV.blockNum = mBlockNum;
                gemmParam4QKxV.biasFloat = reinterpret_cast<float*>(mGemmBias.get());
                gemmParam4QKxV.useInt8 = 0;
                gemmParam4QKxV.fp32minmax = reinterpret_cast<float*>(mGemmRelu.get());
                gemmParam4QKxV.inputScale = (float*)mQKScale.ptr();
                gemmParam4QKxV.inputBias = (float*)mQKBias.ptr();
                gemmParam4QKxV.srcKernelSum = (float*)(mSumQK.ptr() + tId * eP8 * QUANT_INFO_BYTES);

                sumParams4QKxV.oneScale = 0;
                sumParams4QKxV.SRC_UNIT = lP8;
                sumParams4QKxV.blockNum = mBlockNum;
                sumParams4QKxV.DST_XUNIT = eP8;
                sumParams4QKxV.inputBlock = 0;
                sumParams4QKxV.kernelxy = 1;
                sumParams4QKxV.unitColBufferSize = ROUND_UP(MNN_FLASH_ATTENTION_BLOCK_SIZE, lP8) * eP8;
                sumParams4QKxV.kernelCountUnitDouble = UP_DIV(MNN_FLASH_ATTENTION_BLOCK_SIZE, lP8);
            }
        }

        size_t vstride0 = ROUND_UP(mHeadDim, hP) * ROUND_UP(MNN_FLASH_ATTENTION_BLOCK_SIZE, lP);
        if (mQuantValue) {
            vstride0 = (ROUND_UP(mHeadDim, hP8) * ROUND_UP(mKVCacheManager->getFlashAttentionBlockKv(), lP8) + 2 * QUANT_INFO_BYTES * mBlockNum * ROUND_UP(mHeadDim, hP8));
        }

        // use for V
        float const* srcPtr[1];
        // only used for quantized V
        float vQuantScale[1] = {255.f};
        float vQuantBias[1] = {-128.f};
        int32_t infoInt8V[5];
        infoInt8V[0] = 1;       // number
        infoInt8V[2] = static_cast<int32_t>(sumParams4QKxV.unitColBufferSize);
        infoInt8V[3] = 1;       // stride
        int32_t elInt8V[4] = {eP8, ROUND_UP(MNN_FLASH_ATTENTION_BLOCK_SIZE, lP8), 0, 0};

        // only used for float V
        int32_t infoFloatV[4];
        infoFloatV[0] = 1;      // number
        infoFloatV[1] = seqLen; // eReal
        infoFloatV[3] = 1;      // stride
        int32_t elFloatV[4] = {seqLen, ROUND_UP(kvSeqLen, lP), 0, 0};

        int offset[2] = {seqLen, mNumHead * mHeadDim};

        for (int h = headIndex; h < headIndex + headsToCompute; h++) {
            // Prepare for flash attention
            if (runningSum && runningMax) {
                if (sinksPtr == nullptr) {
                    memset(runningSum, 0, mRunningSum->stride(0));
                    for (int k = 0; k < seqLen; ++k) {
                        runningMax[k] = std::numeric_limits<float>::lowest();
                    }
                } else {
                    for (int k = 0; k < seqLen; ++k) {
                        runningSum[k] = 1.f; // exp(sink-sink)
                    }
                    float sinkVal;
                    if (mBytes == 2) {
                        sinkVal = ((FLOAT16_T*)sinksPtr)[h];
                    } else {
                        sinkVal = sinksPtr[h];
                    }
                    for (int k = 0; k < seqLen; ++k) {
                        runningMax[k] = sinkVal;
                    }
                }
            }

            // Compute the current addresses
            int    kvHeadIndex = h / group_size;
            int8_t * keyAddr   = mKVCacheManager->addrOfKey(kvHeadIndex);
            int8_t * keySum    = mKVCacheManager->addrOfKeySum(kvHeadIndex);
            int8_t * valueAddr = mKVCacheManager->addrOfValue(kvHeadIndex);
            float* valueSum    = (float*)mKVCacheManager->addrOfValueSum(kvHeadIndex);

            // Get packed Q
            if (mQuantKey == false) {
                qReordered      = mPackQ->host<int8_t>() + tId * mPackQ->stride(0);
                gcore->MNNAttenPackAndScaleSingleHead((float*)qReordered, (float*)(query->host<int8_t>() + h * mHeadDim * mBytes), mHeadDim * mNumHead, &q_scale, units, seqLen, mHeadDim);
            } else {
                qReordered = mPackQ->host<int8_t>() + h * mPackQ->stride(0);
                qSumAddr = (float*)(mSumQ.ptr() + tId * ROUND_UP(seqLen, eP8) * mBlockNum * QUANT_INFO_BYTES);
                qScale = (float*)(mQueryScale.ptr() + h * seqLen * mBlockNum * QUANT_INFO_BYTES);
                qBias = (float*)(mQueryZeroPoint.ptr() + h * seqLen * mBlockNum * QUANT_INFO_BYTES);
                gcore->MNNSumByAxisLForMatmul_A(qSumAddr, qReordered, qScale, seqLen, sumParams4QxK);
            }

            // Start computing
            for (int i = 0; i < kvBlocks; ++i) {
                int subKvSeqLen = ALIMIN(mBlockKV, kvSeqLen - i * mBlockKV);
                // 1. query @ key
                if (mQuantKey == false) {
                    auto keyPtr = keyAddr + i * UP_DIV(mBlockKV, hP) * ROUND_UP(mHeadDim, lP) * hP * mBytes;
                    int loop_e = seqLen / eP;
                    int remain = seqLen % eP;
                    auto qStride0 = ROUND_UP(mHeadDim, lP) * eP * mBytes;
                    size_t shapeParameters[7] = {(size_t)eP * lP *  mBytes, ROUND_UP((size_t)mHeadDim, lP), (size_t)subKvSeqLen, (size_t)seqLen * mPack * mBytes, 0, 0, 0};
                    for (int ei = 0 ; ei < loop_e; ei++) {
                        gcore->MNNPackedMatMul((float*)(qkPacked + (ei * eP * mPack) * mBytes), (float*)(qReordered + ei * qStride0), (float*)keyPtr, shapeParameters, nullptr, nullptr, nullptr, nullptr);
                    }
                    if (remain > 0) {
                        gcore->MNNPackedMatMulRemain((float*)(qkPacked + (loop_e * eP * mPack) * mBytes), (float*)(qReordered + loop_e * qStride0), (float*)keyPtr, remain, shapeParameters, nullptr, nullptr, nullptr, nullptr);
                    }
                } else {
                    auto eRemain = seqLen;
                    auto srcInt8 = qReordered;
                    auto dstInt8 = qkPacked;
                    auto keyPtr = keyAddr + i * UP_DIV(mBlockKV, hP8) * (ROUND_UP(mHeadDim, lP8) * hP8 + 2 * hP8 * QUANT_INFO_BYTES);
                    gemmParam4QxK.weightKernelSum = (float*)(keySum + i * mBlockKV * QUANT_INFO_BYTES);
                    gemmParam4QxK.inputScale   = qScale;
                    gemmParam4QxK.inputBias    = qBias;
                    gemmParam4QxK.srcKernelSum = qSumAddr;
                    while (eRemain > 0) {
                        auto eSize = ALIMIN(eP8, eRemain);
                        mInt8GemmKernel(dstInt8, srcInt8, keyPtr, UP_DIV(mHeadDim, lP8), mBytes * seqLen * mPack, UP_DIV(subKvSeqLen, mPack), &gemmParam4QxK, eSize);
                        eRemain -= eP8;
                        gemmParam4QxK.inputScale += eP8;
                        gemmParam4QxK.inputBias += eP8;
                        gemmParam4QxK.srcKernelSum += eP8;
                        srcInt8 += unitColBufferSize;
                        dstInt8 += eP8 * mPack * mBytes;
                        if (mBlockNum > 1) {
                            memset(accumbuff, 0, eP8 * hP8 * QUANT_INFO_BYTES);
                            gemmParam4QxK.accumBuffer = accumbuff;
                        }
                    }
                }
                // 2. softmax scores, softmax src/dst shape: [kv_seq_len/mPack, seq_len, mPack]
                {
                    if (mQuantKey == false || isLowerTriangular == false || sinksPtr != nullptr) {
                        if (mBytes == 2) {
                            _maskQK<FLOAT16_T>((float*)qkPacked, &mScale, seqLen, subKvSeqLen, mPack, kvSeqLen, i * mBlockKV, padSeqLength, sinksPtr, mask, mQuantKey, isLowerTriangular);
                        } else {
                            _maskQK<float>((float*)qkPacked, &mScale, seqLen, subKvSeqLen, mPack, kvSeqLen, i * mBlockKV, padSeqLength, sinksPtr, mask, mQuantKey, isLowerTriangular);
                        }
                    }
                    gcore->MNNSoftmax(qkSoftmax, (float*)qkPacked, runningMax, runningSum, diffScale, seqLen, subKvSeqLen, i * mBlockKV, kvValidOffset, mPack, useMaskInSoftmax);
                    if (h2oEnabled && h2oBlockCount > 0) {
                        // NOTE: qkSoftmax is block-local in flash-attention loop.
                        // We intentionally use it as an online heuristic score signal
                        // (not an exact global attention distribution).
                        const int softStride = ROUND_UP(subKvSeqLen, mPack);
                        for (int q = 0; q < seqLen; ++q) {
                            auto row = qkSoftmax + q * softStride;
                            for (int k = 0; k < subKvSeqLen; ++k) {
                                const int globalToken = i * mBlockKV + k;
                                const int blockIndex = globalToken / h2oBlockTokens;
                                if (blockIndex >= 0 && blockIndex < h2oBlockCount) {
                                    localBlockAcc[blockIndex] += row[k];
                                }
                            }
                        }
                    }
                }
                // 3. qk @ v
                auto qkStride0 = ROUND_UP(subKvSeqLen, lP) * eP * mBytes;
                auto rowStart = (!isLowerTriangular || i * mBlockKV < kvValidOffset)? 0 : (i * mBlockKV - kvValidOffset);

                if (mQuantValue == false) {
                    auto valuePtr = valueAddr + i * vstride0 * mBytes;
                    size_t shapeParameters[7] = {(size_t)eP * lP * mBytes, ROUND_UP((size_t)subKvSeqLen, lP), (size_t)mHeadDim, (size_t)seqLen * mPack * mBytes, 0, 0, 0};
                    size_t bExtraStride = (i < kvBlocks - 1) ? 0 : (ROUND_UP(mKVCacheManager->getFlashAttentionBlockKv(), lP) - ROUND_UP(subKvSeqLen, lP)) * hP * mBytes;
                    shapeParameters[5] = bExtraStride;

                    int loop_e = (seqLen - rowStart) / eP;
                    int remain = (seqLen - rowStart) % eP;

                    int ei = 0;
                    elFloatV[0] = eP;
                    elFloatV[1] = ROUND_UP(subKvSeqLen, lP);
                    infoFloatV[2] = eP;
                    for ( ; ei < loop_e; ei++) {
                        srcPtr[0] = (float const*)((int8_t*)qkSoftmax + (ei * eP + rowStart) * mPack * mBytes);
                        gcore->MNNPackC4ForMatMul_A((float*)qkReordered, srcPtr, infoFloatV, elFloatV);
                        gcore->MNNPackedMatMul((float*)(qkvPacked + (ei * eP + rowStart) * mPack * mBytes), (float*)qkReordered, (float*)valuePtr, shapeParameters, nullptr, nullptr, nullptr, nullptr);
                    }
                    if (remain > 0) {
                        elFloatV[0] = remain;
                        infoFloatV[2] = remain;
                        srcPtr[0] = (float const*)((int8_t*)qkSoftmax + (loop_e * eP + rowStart) * mPack * mBytes);
                        shapeParameters[0] = remain * lP * mBytes;
                        gcore->MNNPackC4ForMatMul_A((float*)qkReordered, srcPtr, infoFloatV, elFloatV);
                        gcore->MNNPackedMatMulRemain((float*)(qkvPacked + (loop_e * eP + rowStart) * mPack * mBytes), (float*)qkReordered, (float*)valuePtr, remain, shapeParameters, nullptr, nullptr, nullptr, nullptr);
                    }
                } else { // use int8 kernel to compute qk@ v
                    auto valuePtr = valueAddr + i * vstride0;
                    auto eRemain = seqLen - rowStart;
                    auto qkPtr = (int8_t*)(qkSoftmax) + rowStart * mPack * mBytes; // [UP_DIV(subKvSeqLen,pack),seqLen,pack]
                    auto qkvFloat = qkvPacked + rowStart * mPack * mBytes;
                    gemmParam4QKxV.weightKernelSum = valueSum + i * ROUND_UP(mHeadDim, hP8);
                    sumParams4QKxV.valid = subKvSeqLen % lP8;
                    sumParams4QKxV.LU = UP_DIV(subKvSeqLen, lP8);

                    auto dstInt8Ptr = (int8_t*)mQuantQK.ptr() + tId * eP8 * ROUND_UP(MNN_FLASH_ATTENTION_BLOCK_SIZE, mPack);
                    srcPtr[0] = (const float*)(dstInt8Ptr);

                    while (eRemain > 0) {
                        auto eSize = ALIMIN(eRemain, eP8);

                        memset(dstInt8Ptr, 0, eP8 * ROUND_UP(MNN_FLASH_ATTENTION_BLOCK_SIZE, mPack));

                        infoInt8V[1] = eSize; // eReal
                        infoInt8V[4] = eSize; // e to process
                        elInt8V[0] = eSize;   // e to process


                        for (int qi = 0; qi < UP_DIV(subKvSeqLen, mPack); ++qi) {
                            mQuantFunc((float*)(qkPtr + qi * seqLen * mPack * mBytes), dstInt8Ptr + qi * eSize * mPack, eSize, vQuantScale, -128, 127, vQuantBias, 0);
                        }
                        core->MNNPackC4Int8ForMatMul_A(qkReordered, (int8_t const **)srcPtr, infoInt8V, elInt8V);
                        // mSumQK
                        gcore->MNNSumByAxisLForMatmul_A(gemmParam4QKxV.srcKernelSum, qkReordered, (float*)mQKScale.ptr(), eSize, sumParams4QKxV);
                        mInt8GemmKernel(qkvFloat, qkReordered, valuePtr, UP_DIV(MNN_FLASH_ATTENTION_BLOCK_SIZE, lP8), mBytes * seqLen * mPack, UP_DIV(mHeadDim, mPack), &gemmParam4QKxV, eSize);

                        eRemain -= eSize;
                        qkPtr += (eSize * mPack * mBytes);
                        qkvFloat += (eSize * mPack * mBytes);
                    }
                }

                // 4. flash attention, update each sub kvSeq's final results
                if (runningMax != nullptr && runningSum != nullptr && diffScale != nullptr) {
                    gcore->MNNFlashAttentionUpdateBlockOutput((float*)outputPacked, (float*)qkvPacked, diffScale, runningSum, UP_DIV(mHeadDim, mPack), seqLen, mPack, i, kvBlocks, mPackQKV->stride(0) / mBytes, mBytes, rowStart);
                }
            }

            // Final results writing: [head_dim/mPack, seq_len, mPack] -> [seq_len, num_head, head_dim]
            auto dstPtr = outputs[0]->host<int8_t>() + h * mHeadDim * mBytes;
            // offset = {seqLen, mNumHead * mHeadDim};
            gcore->MNNUnpackCUnitTranspose((float*)dstPtr, (float*)outputPacked, seqLen, mHeadDim, offset);
        }
        if (h2oEnabled && h2oBlockCount > 0) {
            std::lock_guard<std::mutex> guard(h2oBlockLock);
            for (int i = 0; i < h2oBlockCount; ++i) {
                h2oBlockAcc[i] += localBlockAcc[i];
            }
        }
    };

    MNN_CONCURRENCY_BEGIN(tId, mThreadNum) {
        mCompute((int)tId);
    }
    MNN_CONCURRENCY_END();

    int finalKeepTokensForLayer = kvSeqLen;
    CPUAttention::H2OSharedState::LayerState* scoreStatePtr = nullptr;
    if (mMeta != nullptr && mMeta->h2o_in_decode != 0 && mH2OState != nullptr) {
        // Keep one global score state so trigger cadence is stable even if runtime
        // does not expose per-layer execution boundaries.
        if (mH2OState->layerStates.empty()) {
            mH2OState->layerStates.resize(1);
            mH2OState->layerCacheManagers.resize(1, nullptr);
        }
        scoreStatePtr = &mH2OState->layerStates[0];
    }

    if (h2oEnabled && h2oBlockCount > 0 && mMeta != nullptr && scoreStatePtr != nullptr) {
        auto& scoreState = *scoreStatePtr;
        if ((int)scoreState.blockScores.size() != h2oBlockCount) {
            scoreState.blockScores.resize(h2oBlockCount, 0.0f);
        }

        const float alpha = ALIMIN(1.0f, ALIMAX(0.0f, mMeta->h2o_ema_alpha));
        const float norm = (float)ALIMAX(1, mNumHead * seqLen);
        for (int i = 0; i < h2oBlockCount; ++i) {
            const float current = h2oBlockAcc[i] / norm;
            scoreState.blockScores[i] = alpha * scoreState.blockScores[i] + (1.0f - alpha) * current;
        }

        const int triggerMin = ALIMAX(1, mMeta->h2o_trigger_min_tokens);
        const int updateInterval = ALIMAX(1, mMeta->h2o_update_interval);
        const bool shouldTrigger = kvSeqLen >= triggerMin
            && (mH2OState->globalStep - mH2OState->globalLastTriggerStep >= updateInterval);

        if (shouldTrigger) {
            mH2OState->globalLastTriggerStep = mH2OState->globalStep;
            auto t0 = std::chrono::high_resolution_clock::now();

            std::vector<char> keepBlock(h2oBlockCount, 0);
            const int sinkTokens = ALIMAX(0, mMeta->h2o_sink_tokens);
            const int recentTokens = ALIMAX(0, mMeta->h2o_recent_tokens);
            const int recentStartToken = ALIMAX(0, kvSeqLen - recentTokens);
            const int sinkEndToken = ALIMIN(kvSeqLen, sinkTokens);

            auto blockTokenSize = [=](int blockIndex) {
                const int start = blockIndex * h2oBlockTokens;
                const int end = ALIMIN(kvSeqLen, start + h2oBlockTokens);
                return ALIMAX(0, end - start);
            };

            for (int token = 0; token < sinkEndToken; token += h2oBlockTokens) {
                keepBlock[token / h2oBlockTokens] = 1;
            }
            for (int block = recentStartToken / h2oBlockTokens; block < h2oBlockCount; ++block) {
                keepBlock[block] = 1;
            }

            int keptTokens = 0;
            for (int i = 0; i < h2oBlockCount; ++i) {
                if (keepBlock[i]) {
                    keptTokens += blockTokenSize(i);
                }
            }
            mMeta->h2o_floor_keep_by_recent_sink = kvSeqLen > 0 ? (float)keptTokens / (float)kvSeqLen : 1.0f;

            float targetKeepRatio = ALIMIN(1.0f, ALIMAX(0.0f, mMeta->h2o_target_keep_ratio));
            if (mMeta->h2o_target_mode != 0) {
                const float targetLossy = ALIMAX(1.0f, mMeta->h2o_target_lossy_ratio);
                targetKeepRatio = 1.0f / targetLossy;
            }
            int targetKeep = (int)std::ceil(targetKeepRatio * kvSeqLen);
            targetKeep = ALIMAX(1, ALIMIN(kvSeqLen, targetKeep));
            const int targetKeepBeforeFloor = targetKeep;
            targetKeep = ALIMAX(targetKeep, keptTokens);
            // When sink+recent floor already meets the original target, any
            // evicted tokens are non-selected middle tokens — blind eviction
            // that destroys prompt context without quality-aware selection.
            const bool floorDominatesTarget = (keptTokens >= targetKeepBeforeFloor);
            mMeta->h2o_target_keep_effective = kvSeqLen > 0 ? (float)targetKeep / (float)kvSeqLen : 1.0f;
            if (mMeta->h2o_log_stats != 0 && keptTokens > targetKeepBeforeFloor) {
                MNN_PRINT("[H2O] floor-keep dominates target: kv=%d target=%d floor=%d (sink=%d recent=%d block=%d)\n",
                          kvSeqLen,
                          targetKeepBeforeFloor,
                          keptTokens,
                          sinkTokens,
                          recentTokens,
                          h2oBlockTokens);
            }

            int quantizedTargetKeep = keptTokens;
            if (quantizedTargetKeep < targetKeep) {
                const int needTokens = targetKeep - quantizedTargetKeep;
                const int needBlocks = UP_DIV(needTokens, h2oBlockTokens);
                quantizedTargetKeep = ALIMIN(kvSeqLen, quantizedTargetKeep + needBlocks * h2oBlockTokens);
            }
            mMeta->h2o_block_quantized_keep = kvSeqLen > 0 ? (float)quantizedTargetKeep / (float)kvSeqLen : 1.0f;

            if (keptTokens < quantizedTargetKeep) {
                std::vector<std::pair<float, int>> candidates;
                candidates.reserve(h2oBlockCount);
                for (int i = 0; i < h2oBlockCount; ++i) {
                    if (!keepBlock[i]) {
                        candidates.emplace_back(scoreState.blockScores[i], i);
                    }
                }
                std::sort(candidates.begin(), candidates.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                    return a.first > b.first;
                });
                for (auto& item : candidates) {
                    if (keptTokens >= quantizedTargetKeep) {
                        break;
                    }
                    keepBlock[item.second] = 1;
                    keptTokens += blockTokenSize(item.second);
                }
            }

            std::vector<int> reservePairs;
            reservePairs.reserve(h2oBlockCount * 2);
            int runBegin = -1;
            int runEnd = -1;
            for (int i = 0; i < h2oBlockCount; ++i) {
                if (!keepBlock[i]) {
                    continue;
                }
                const int blockBegin = i * h2oBlockTokens;
                const int blockEnd = ALIMIN(kvSeqLen, blockBegin + h2oBlockTokens);
                if (runBegin < 0) {
                    runBegin = blockBegin;
                    runEnd = blockEnd;
                } else if (blockBegin == runEnd) {
                    runEnd = blockEnd;
                } else {
                    reservePairs.emplace_back(runBegin);
                    reservePairs.emplace_back(runEnd - runBegin);
                    runBegin = blockBegin;
                    runEnd = blockEnd;
                }
            }
            if (runBegin >= 0) {
                reservePairs.emplace_back(runBegin);
                reservePairs.emplace_back(runEnd - runBegin);
            }

            int finalKeepTokens = 0;
            for (size_t i = 0; i + 1 < reservePairs.size(); i += 2) {
                finalKeepTokens += reservePairs[i + 1];
            }
            finalKeepTokensForLayer = finalKeepTokens;
            // Skip eviction when floor dominates: the protected zones (sink +
            // recent) already satisfy the compression target, so there is no
            // score-based selection.  Evicting the remaining middle tokens
            // would blindly destroy prompt context.
            const int evictedTokens = floorDominatesTarget
                ? 0
                : ALIMAX(0, kvSeqLen - finalKeepTokens);
            const size_t reserveMetaBytes = (size_t)reservePairs.size() * sizeof(int);

            if (mH2OState != nullptr
                && layerIndex >= 0
                && layerIndex < (int)mH2OState->layerStates.size()) {
                auto& pendingState = mH2OState->layerStates[layerIndex];
                if (evictedTokens > 0 && !reservePairs.empty()) {
                    pendingState.pendingReservePairs.swap(reservePairs);
                    pendingState.pendingRemove = (size_t)kvSeqLen;
                    pendingState.pendingPlanReady = 1;
                    mMeta->h2o_pending_remove = pendingState.pendingRemove;
                    mMeta->h2o_pending_reserve = nullptr;
                    mMeta->h2o_pending_n_reserve = (int)pendingState.pendingReservePairs.size() / 2;
                    mMeta->h2o_pending_plan_ready = 1;
                    mMeta->h2o_total_evict_tokens += evictedTokens;
                } else {
                    pendingState.pendingPlanReady = 0;
                    pendingState.pendingRemove = 0;
                    pendingState.pendingReservePairs.clear();
                    mMeta->h2o_pending_plan_ready = 0;
                    mMeta->h2o_pending_remove = 0;
                    mMeta->h2o_pending_reserve = nullptr;
                    mMeta->h2o_pending_n_reserve = 0;
                }
            }

            const size_t bytesPerToken = (size_t)mKvNumHead * (size_t)mHeadDim * (size_t)mBytes * 2;
            const size_t rawBefore = (size_t)kvSeqLen * bytesPerToken;
            const size_t rawAfter = (size_t)finalKeepTokens * bytesPerToken;
            const size_t lossyDen = rawAfter + reserveMetaBytes;
            mMeta->h2o_keep_ratio = kvSeqLen > 0 ? (float)finalKeepTokens / (float)kvSeqLen : 1.0f;
            mMeta->h2o_lossy_ratio = lossyDen > 0 ? (float)rawBefore / (float)lossyDen : 1.0f;
            mMeta->h2o_last_evict_tokens = evictedTokens;

            if (evictedTokens > 0) {
                // KV compaction renumbers kept blocks; remap EMA scores to the
                // compacted order so next-step ranking is aligned.
                // NOTE: The remapped count (kept blocks) may differ from the new
                // h2oBlockCount after compaction, because partial blocks from
                // different kept ranges can merge under new block boundaries.
                // The resize at line ~2049 handles the size mismatch (truncation
                // or zero-extension), and EMA alpha=0.9 self-corrects within a
                // few steps, so this approximation is acceptable.
                std::vector<float> remappedScores;
                remappedScores.reserve(h2oBlockCount);
                for (int i = 0; i < h2oBlockCount; ++i) {
                    if (keepBlock[i]) {
                        remappedScores.emplace_back(scoreState.blockScores[i]);
                    }
                }
                if (!remappedScores.empty()) {
                    scoreState.blockScores.swap(remappedScores);
                } else {
                    scoreState.blockScores.clear();
                }
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            mMeta->h2o_evict_us = (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            if (mMeta->h2o_log_stats != 0 && mMeta->h2o_last_evict_tokens > 0) {
                MNN_PRINT("[H2O] layer=%d kv=%d keep_ratio=%.4f evict=%d lossy_ratio=%.4f evict_us=%lld reserve_pairs=%d target=%.3f effective=%.3f floor=%.3f quantized=%.3f recent=%d sink=%d block=%d\n",
                          layerIndex,
                          kvSeqLen,
                          mMeta->h2o_keep_ratio,
                          mMeta->h2o_last_evict_tokens,
                          mMeta->h2o_lossy_ratio,
                          (long long)mMeta->h2o_evict_us,
                          mMeta->h2o_pending_n_reserve,
                          mMeta->h2o_target_keep_ratio,
                          mMeta->h2o_target_keep_effective,
                          mMeta->h2o_floor_keep_by_recent_sink,
                          mMeta->h2o_block_quantized_keep,
                          mMeta->h2o_recent_tokens,
                          mMeta->h2o_sink_tokens,
                          h2oBlockTokens);
            }
        } else if (kvSeqLen > 0) {
            finalKeepTokensForLayer = ALIMAX(1, (int)std::round(mMeta->h2o_keep_ratio * kvSeqLen));
        }
    }

    if (mMeta != nullptr && mMeta->h2o_in_decode != 0 && mH2OState != nullptr) {
        struct LosslessRuntimeStats {
            float ratio = 1.0f;
            float attemptedRatio = 1.0f;
            uint64_t rawBytes = 0;
            uint64_t compressedBytes = 0;
            uint64_t attemptedCompressedBytes = 0;
            uint64_t decompressedBytes = 0;
            int64_t compressUs = 0;
            int64_t decompressUs = 0;
            bool fallbackUsed = false;
            bool backpressureSkipped = false;
            uint64_t rawHash = 0;
            uint64_t blobHash = 0;
            std::vector<uint8_t> keyBlob;
            std::vector<uint8_t> valueBlob;
        };
        struct LosslessCollectPayload {
            int evalStart = 0;
            int evalTokens = 0;
            uint64_t rawBytes = 0;
            uint64_t rawHash = 0;
            bool fallbackUsed = false;
            std::vector<uint8_t> keyMerged;
            std::vector<uint8_t> valueMerged;
        };
        const PredictorFlags keyFlags = parsePredictorFlags(mMeta->h2o_lossless_predictors_k, true);
        const PredictorFlags valueFlags = parsePredictorFlags(mMeta->h2o_lossless_predictors_v, true);
        const bool strictRoundtrip = (mMeta->h2o_lossless_strict_roundtrip_check != 0);
        auto collectLosslessRuntimePayload = [&](int startToken, int tokenCount) -> LosslessCollectPayload {
            LosslessCollectPayload payload;
            if (mMeta->h2o_lossless_codec != 1
                || mMeta->h2o_lossless_runtime_enable == 0
                || tokenCount <= 0
                || kvSeqLen <= 0) {
                return payload;
            }
            if (mBytes != 2 && mBytes != 4) {
                payload.fallbackUsed = true;
                return payload;
            }
            payload.evalStart = clampInt(startToken, 0, kvSeqLen);
            payload.evalTokens = clampInt(tokenCount, 0, kvSeqLen - payload.evalStart);
            if (payload.evalTokens <= 0) {
                return payload;
            }
            const int flashBlockKv = ALIMAX(1, mKVCacheManager->getFlashAttentionBlockKv());
            if (!collectKvMergedRange(
                    mKVCacheManager.get(),
                    mKvNumHead,
                    mHeadDim,
                    mBytes,
                    lP,
                    hP,
                    flashBlockKv,
                    payload.evalStart,
                    payload.evalTokens,
                    payload.keyMerged,
                    payload.valueMerged)) {
                payload.fallbackUsed = true;
                payload.keyMerged.clear();
                payload.valueMerged.clear();
                return payload;
            }
            payload.rawBytes = (uint64_t)payload.keyMerged.size() + (uint64_t)payload.valueMerged.size();
            if (payload.rawBytes == 0) {
                // Empty gather is a no-op sample (no bytes to encode), not a codec failure.
                // Keep fallback=false to avoid tripping runtime fallback gate on benign skips.
                payload.fallbackUsed = false;
                return payload;
            }
            if (strictRoundtrip) {
                const uint64_t keyHash = fnv1a64(payload.keyMerged.data(), payload.keyMerged.size());
                payload.rawHash = fnv1a64(payload.valueMerged.data(), payload.valueMerged.size(), keyHash);
            }
            return payload;
        };
        auto encodeLosslessRuntimePayload = [&](const LosslessCollectPayload& payload) -> LosslessRuntimeStats {
            LosslessRuntimeStats stats;
            if (payload.fallbackUsed || payload.rawBytes == 0) {
                stats.fallbackUsed = payload.fallbackUsed;
                return stats;
            }
            stats.rawBytes = payload.rawBytes;
            stats.rawHash = payload.rawHash;
            auto c0 = std::chrono::high_resolution_clock::now();
            bool keyOk = false;
            bool valueOk = false;
            if (!payload.keyMerged.empty()) {
                if (mBytes == 2) {
                    keyOk = encodeFp16GearPredictive(payload.keyMerged.data(), payload.keyMerged.size(), keyFlags, keyFlags, stats.keyBlob);
                } else {
                    keyOk = encodeFp32LanePredictive(payload.keyMerged.data(), payload.keyMerged.size(), keyFlags, stats.keyBlob);
                }
            } else {
                keyOk = true;
            }
            if (!payload.valueMerged.empty()) {
                if (mBytes == 2) {
                    valueOk = encodeFp16GearPredictive(payload.valueMerged.data(), payload.valueMerged.size(), valueFlags, valueFlags, stats.valueBlob);
                } else {
                    valueOk = encodeFp32LanePredictive(payload.valueMerged.data(), payload.valueMerged.size(), valueFlags, stats.valueBlob);
                }
            } else {
                valueOk = true;
            }
            auto c1 = std::chrono::high_resolution_clock::now();
            stats.compressUs = (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(c1 - c0).count();

            if (!keyOk || !valueOk) {
                stats.fallbackUsed = true;
                stats.keyBlob.clear();
                stats.valueBlob.clear();
                stats.compressedBytes = stats.rawBytes;
                stats.attemptedCompressedBytes = stats.rawBytes;
                stats.attemptedRatio = 1.0f;
                stats.ratio = 1.0f;
                return stats;
            }
            const uint64_t attemptedBytes = static_cast<uint64_t>(stats.keyBlob.size() + stats.valueBlob.size());
            if (attemptedBytes == 0) {
                stats.fallbackUsed = true;
                stats.keyBlob.clear();
                stats.valueBlob.clear();
                stats.compressedBytes = stats.rawBytes;
                stats.attemptedCompressedBytes = stats.rawBytes;
                stats.attemptedRatio = 1.0f;
                stats.ratio = 1.0f;
                return stats;
            }
            stats.attemptedCompressedBytes = attemptedBytes;
            stats.attemptedRatio = static_cast<float>((double)payload.rawBytes / (double)attemptedBytes);
            if (attemptedBytes >= payload.rawBytes) {
                // Compression is not beneficial for this block; treat as a normal no-gain path.
                // Keep fallback=false so fallback metrics reflect actual runtime errors only.
                stats.fallbackUsed = false;
                stats.keyBlob.clear();
                stats.valueBlob.clear();
                stats.compressedBytes = stats.rawBytes;
                stats.ratio = 1.0f;
                return stats;
            }
            stats.compressedBytes = attemptedBytes;
            stats.ratio = static_cast<float>((double)payload.rawBytes / (double)attemptedBytes);
            stats.blobHash = fnv1a64(stats.valueBlob.data(), stats.valueBlob.size(),
                                     fnv1a64(stats.keyBlob.data(), stats.keyBlob.size()));
            return stats;
        };
        auto encodeLosslessRuntimeStats = [&](int startToken, int tokenCount) -> LosslessRuntimeStats {
            auto payload = collectLosslessRuntimePayload(startToken, tokenCount);
            if (payload.fallbackUsed && payload.rawBytes == 0) {
                LosslessRuntimeStats stats;
                stats.fallbackUsed = true;
                return stats;
            }
            return encodeLosslessRuntimePayload(payload);
        };

        auto decodeOnePendingLosslessBlock = [&](CPUAttention::H2OSharedState::LayerState& layerState) {
            const int decodeCacheBlocks = ALIMAX(0, mMeta->h2o_lossless_decode_cache_blocks);
            const int globalDecodeCacheBlocks = decodeCacheBlocks;
            if (decodeCacheBlocks <= 0) {
                layerState.decodeCacheEntries.clear();
                mH2OState->globalDecodeCacheEntries.clear();
            }
            auto appendLocalDecodeCache = [&](int startToken,
                                              int tokenCount,
                                              uint64_t rawHash,
                                              uint64_t blobHash,
                                              uint64_t rawBytes,
                                              uint64_t compressedBytes,
                                              const std::vector<uint8_t>& keyDecoded,
                                              const std::vector<uint8_t>& valueDecoded) {
                if (decodeCacheBlocks <= 0) {
                    return;
                }
                for (auto cIt = layerState.decodeCacheEntries.begin();
                     cIt != layerState.decodeCacheEntries.end();) {
                    const bool posMatch = cIt->startToken == startToken && cIt->tokenCount == tokenCount;
                    const bool bytesMatch = cIt->rawBytes == rawBytes && cIt->compressedBytes == compressedBytes;
                    const bool hashMatch = (rawHash != 0 && cIt->rawHash == rawHash)
                        || (blobHash != 0 && cIt->blobHash == blobHash);
                    if ((posMatch && bytesMatch) || (bytesMatch && hashMatch)) {
                        cIt = layerState.decodeCacheEntries.erase(cIt);
                    } else {
                        ++cIt;
                    }
                }
                CPUAttention::H2OSharedState::LayerState::DecodedCacheEntry entry;
                entry.startToken = startToken;
                entry.tokenCount = tokenCount;
                entry.rawHash = rawHash;
                entry.blobHash = blobHash;
                entry.rawBytes = rawBytes;
                entry.compressedBytes = compressedBytes;
                entry.keyDecoded = keyDecoded;
                entry.valueDecoded = valueDecoded;
                layerState.decodeCacheEntries.push_back(std::move(entry));
                while ((int)layerState.decodeCacheEntries.size() > decodeCacheBlocks) {
                    layerState.decodeCacheEntries.pop_front();
                }
            };
            auto appendGlobalDecodeCache = [&](int startToken,
                                               int tokenCount,
                                               uint64_t rawHash,
                                               uint64_t blobHash,
                                               uint64_t rawBytes,
                                               uint64_t compressedBytes,
                                               const std::vector<uint8_t>& keyDecoded,
                                               const std::vector<uint8_t>& valueDecoded) {
                if (globalDecodeCacheBlocks <= 0) {
                    return;
                }
                auto& globalCache = mH2OState->globalDecodeCacheEntries;
                for (auto cIt = globalCache.begin(); cIt != globalCache.end();) {
                    const bool bytesMatch = cIt->rawBytes == rawBytes && cIt->compressedBytes == compressedBytes;
                    const bool hashMatch = (rawHash != 0 && cIt->rawHash == rawHash)
                        || (blobHash != 0 && cIt->blobHash == blobHash);
                    if (bytesMatch && hashMatch) {
                        cIt = globalCache.erase(cIt);
                    } else {
                        ++cIt;
                    }
                }
                CPUAttention::H2OSharedState::LayerState::DecodedCacheEntry entry;
                entry.startToken = startToken;
                entry.tokenCount = tokenCount;
                entry.rawHash = rawHash;
                entry.blobHash = blobHash;
                entry.rawBytes = rawBytes;
                entry.compressedBytes = compressedBytes;
                entry.keyDecoded = keyDecoded;
                entry.valueDecoded = valueDecoded;
                globalCache.push_back(std::move(entry));
                while ((int)globalCache.size() > globalDecodeCacheBlocks) {
                    globalCache.pop_front();
                }
            };
            for (auto& block : layerState.losslessBlocks) {
                if (block.decodedOnce) {
                    continue;
                }
                block.decodedOnce = true;
                if (block.keyBlob.empty() && block.valueBlob.empty()) {
                    layerState.losslessFallbackCount += 1;
                    return;
                }
                const uint64_t blockBlobHash = (block.blobHash != 0)
                    ? block.blobHash
                    : fnv1a64(block.valueBlob.data(), block.valueBlob.size(),
                              fnv1a64(block.keyBlob.data(), block.keyBlob.size()));
                std::vector<uint8_t> keyDecoded;
                std::vector<uint8_t> valueDecoded;
                bool cacheHit = false;
                auto tryDecodeCache = [&](bool requirePosShape) {
                    for (auto it = layerState.decodeCacheEntries.begin(); it != layerState.decodeCacheEntries.end(); ++it) {
                        if (requirePosShape) {
                            const bool shapeMatch = it->startToken == block.startToken
                                && it->tokenCount == block.tokenCount;
                            if (!shapeMatch) {
                                continue;
                            }
                        }
                        const bool bytesMatch = it->rawBytes == block.rawBytes
                            && it->compressedBytes == block.compressedBytes;
                        if (!bytesMatch) {
                            continue;
                        }
                        const bool hashMatch = (block.rawHash != 0 && it->rawHash == block.rawHash)
                            || (blockBlobHash != 0 && it->blobHash == blockBlobHash);
                        if (!hashMatch) {
                            continue;
                        }
                        keyDecoded = it->keyDecoded;
                        valueDecoded = it->valueDecoded;
                        auto entry = std::move(*it);
                        layerState.decodeCacheEntries.erase(it);
                        layerState.decodeCacheEntries.push_back(std::move(entry));
                        cacheHit = true;
                        mH2OState->globalLosslessDecodeCacheHit += 1;
                        return;
                    }
                };
                tryDecodeCache(true);
                if (!cacheHit) {
                    tryDecodeCache(false);
                }
                if (!cacheHit && globalDecodeCacheBlocks > 0) {
                    auto& globalCache = mH2OState->globalDecodeCacheEntries;
                    for (auto cIt = globalCache.begin(); cIt != globalCache.end(); ++cIt) {
                        const bool bytesMatch = cIt->rawBytes == block.rawBytes
                            && cIt->compressedBytes == block.compressedBytes;
                        if (!bytesMatch) {
                            continue;
                        }
                        const bool hashMatch = (block.rawHash != 0 && cIt->rawHash == block.rawHash)
                            || (blockBlobHash != 0 && cIt->blobHash == blockBlobHash);
                        if (!hashMatch) {
                            continue;
                        }
                        keyDecoded = cIt->keyDecoded;
                        valueDecoded = cIt->valueDecoded;
                        auto entry = std::move(*cIt);
                        globalCache.erase(cIt);
                        globalCache.push_back(std::move(entry));
                        cacheHit = true;
                        mH2OState->globalLosslessDecodeCacheHit += 1;
                        appendLocalDecodeCache(
                            block.startToken,
                            block.tokenCount,
                            block.rawHash,
                            blockBlobHash,
                            block.rawBytes,
                            block.compressedBytes,
                            keyDecoded,
                            valueDecoded);
                        break;
                    }
                }
                if (!cacheHit) {
                    auto d0 = std::chrono::high_resolution_clock::now();
                    bool keyOk = true;
                    bool valueOk = true;
                    if (!block.keyBlob.empty()) {
                        if (mBytes == 2) {
                            keyOk = decodeFp16GearPredictive(block.keyBlob.data(), block.keyBlob.size(), keyDecoded);
                        } else {
                            keyOk = decodeFp32LanePredictive(block.keyBlob.data(), block.keyBlob.size(), keyDecoded);
                        }
                    }
                    if (!block.valueBlob.empty()) {
                        if (mBytes == 2) {
                            valueOk = decodeFp16GearPredictive(block.valueBlob.data(), block.valueBlob.size(), valueDecoded);
                        } else {
                            valueOk = decodeFp32LanePredictive(block.valueBlob.data(), block.valueBlob.size(), valueDecoded);
                        }
                    }
                    auto d1 = std::chrono::high_resolution_clock::now();
                    const int64_t decodeUs = (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(d1 - d0).count();
                    // Count real decode work even if round-trip validation fails.
                    layerState.losslessDecompressUs += decodeUs;
                    mH2OState->globalLosslessDecodeCacheMiss += 1;
                    if (!keyOk || !valueOk) {
                        layerState.losslessFallbackCount += 1;
                        return;
                    }
                }
                const uint64_t decodedBytes = static_cast<uint64_t>(keyDecoded.size() + valueDecoded.size());
                if (decodedBytes != block.rawBytes) {
                    layerState.losslessFallbackCount += 1;
                    return;
                }
                if (mMeta->h2o_lossless_strict_roundtrip_check != 0) {
                    uint64_t decodedHash = fnv1a64(keyDecoded.data(), keyDecoded.size());
                    decodedHash = fnv1a64(valueDecoded.data(), valueDecoded.size(), decodedHash);
                    if (decodedHash != block.rawHash) {
                        layerState.losslessFallbackCount += 1;
                        return;
                    }
                }
                if (!cacheHit) {
                    appendLocalDecodeCache(
                        block.startToken,
                        block.tokenCount,
                        block.rawHash,
                        blockBlobHash,
                        block.rawBytes,
                        block.compressedBytes,
                        keyDecoded,
                        valueDecoded);
                }
                appendGlobalDecodeCache(
                    block.startToken,
                    block.tokenCount,
                    block.rawHash,
                    blockBlobHash,
                    block.rawBytes,
                    block.compressedBytes,
                    keyDecoded,
                    valueDecoded);
                layerState.losslessDecompressedBytes += decodedBytes;
                return;
            }
        };

        bool applyLossless = false;
        int losslessTokenBudget = kvSeqLen;
        const int runtimeMode = mMeta->h2o_lossless_runtime_mode;
        const bool runtimeProbeMode = (runtimeMode == 0);
        const bool runtimeStoreModeLocal = (runtimeMode == 2);
        const bool asyncEncodeEnabled =
            (mMeta->h2o_lossless_async_threads > 0) && !runtimeProbeMode;
        auto getAsyncPendingCount = [&]() -> int64_t {
            if (mH2OState == nullptr) {
                return 0;
            }
            std::lock_guard<std::mutex> lock(mH2OState->asyncMutex);
            return mH2OState->asyncPendingTasks;
        };
        auto applyEncodedStatsToLayer =
            [&](int targetLayerIndex,
                int startToken,
                int tokenCount,
                LosslessRuntimeStats stats,
                bool updateSchedule,
                bool bootstrapWindowSampled,
                int coldTokenBudget,
                int64_t scheduleStep) {
                if (targetLayerIndex < 0 || targetLayerIndex >= (int)mH2OState->layerStates.size()) {
                    return;
                }
                auto& targetLayerState = mH2OState->layerStates[targetLayerIndex];
                CPUKVCacheManager* targetCacheManager = nullptr;
                if (targetLayerIndex >= 0
                    && targetLayerIndex < (int)mH2OState->layerCacheManagers.size()) {
                    targetCacheManager = mH2OState->layerCacheManagers[targetLayerIndex];
                }
                if (targetCacheManager == nullptr) {
                    targetLayerState.losslessFallbackCount += 1;
                    MNN_ERROR("[H2O-LOSSLESS] missing target KV cache manager: target_layer=%d current_layer=%d\n",
                              targetLayerIndex, layerIndex);
                    return;
                }
                const int targetFlashBlockKv = ALIMAX(1, (int)targetCacheManager->getFlashAttentionBlockKv());
                auto blockOrigin = CPUAttention::H2OSharedState::LayerState::LosslessBlock::FROM_RAW_KV;
                bool countRatioStats = true;
                if (runtimeStoreModeLocal && tokenCount > 0) {
                    const int64_t rangeBegin = static_cast<int64_t>(startToken);
                    const int64_t rangeEnd = rangeBegin + static_cast<int64_t>(tokenCount);
                    for (const auto& existing : targetLayerState.losslessBlocks) {
                        if (!existing.rawDropped) {
                            continue;
                        }
                        const int64_t existingBegin = static_cast<int64_t>(existing.startToken);
                        const int64_t existingEnd = existingBegin + static_cast<int64_t>(existing.tokenCount);
                        const bool overlap = !(rangeEnd <= existingBegin || existingEnd <= rangeBegin);
                        if (overlap) {
                            blockOrigin = CPUAttention::H2OSharedState::LayerState::LosslessBlock::FROM_DROPPED_KV;
                            // In store mode, once raw KV has been dropped for this range,
                            // follow-up re-encodes are only for reuse and should not
                            // inflate lossless ratio accounting.
                            countRatioStats = false;
                            break;
                        }
                    }
                }
                if (!stats.fallbackUsed && (!stats.keyBlob.empty() || !stats.valueBlob.empty())) {
                    const int maxLen = targetCacheManager->maxLength();
                    const int endToken = startToken + tokenCount;
                    if (startToken < 0 || tokenCount <= 0 || endToken < startToken || endToken > maxLen) {
                        targetLayerState.losslessFallbackCount += 1;
                        MNN_ERROR("[H2O-LOSSLESS] invalid encoded range: target_layer=%d current_layer=%d start=%d tokens=%d end=%d max=%d\n",
                                  targetLayerIndex, layerIndex, startToken, tokenCount, endToken, maxLen);
                        return;
                    }
                    CPUAttention::H2OSharedState::LayerState::LosslessBlock block;
                    block.startToken = startToken;
                    block.tokenCount = tokenCount;
                    block.rawBytes = stats.rawBytes;
                    block.compressedBytes = stats.compressedBytes;
                    block.rawHash = stats.rawHash;
                    block.blobHash = stats.blobHash;
                    block.decodedOnce = runtimeStoreModeLocal;
                    block.origin = blockOrigin;
                    block.keyBlob.swap(stats.keyBlob);
                    block.valueBlob.swap(stats.valueBlob);
                    if (block.blobHash == 0 && (!block.keyBlob.empty() || !block.valueBlob.empty())) {
                        block.blobHash = fnv1a64(block.valueBlob.data(), block.valueBlob.size(),
                                                 fnv1a64(block.keyBlob.data(), block.keyBlob.size()));
                    }
                    targetLayerState.losslessBlocks.emplace_back(std::move(block));

                    const int decodeCacheBlocks = ALIMAX(0, mMeta->h2o_lossless_decode_cache_blocks);
                    if (decodeCacheBlocks > 0 && !runtimeStoreModeLocal) {
                        while ((int)targetLayerState.losslessBlocks.size() > decodeCacheBlocks) {
                            targetLayerState.losslessBlocks.erase(targetLayerState.losslessBlocks.begin());
                        }
                    }
                    if (runtimeStoreModeLocal && !targetLayerState.losslessBlocks.empty()) {
                        auto& stored = targetLayerState.losslessBlocks.back();
                        if (decodeCacheBlocks > 0) {
                            std::vector<uint8_t> seededKey;
                            std::vector<uint8_t> seededValue;
                            const bool seededOk = collectKvMergedRange(
                                targetCacheManager,
                                mKvNumHead,
                                mHeadDim,
                                mBytes,
                                lP,
                                hP,
                                targetFlashBlockKv,
                                stored.startToken,
                                stored.tokenCount,
                                seededKey,
                                seededValue);
                            if (seededOk) {
                                const uint64_t seededBytes = (uint64_t)seededKey.size() + (uint64_t)seededValue.size();
                                if (seededBytes == stored.rawBytes) {
                                    for (auto it = targetLayerState.decodeCacheEntries.begin();
                                         it != targetLayerState.decodeCacheEntries.end();) {
                                        const bool posMatch = it->startToken == stored.startToken
                                            && it->tokenCount == stored.tokenCount;
                                        const bool bytesMatch = it->rawBytes == stored.rawBytes
                                            && it->compressedBytes == stored.compressedBytes;
                                        const bool hashMatch = (stored.rawHash != 0 && it->rawHash == stored.rawHash)
                                            || (stored.blobHash != 0 && it->blobHash == stored.blobHash);
                                        if ((posMatch && bytesMatch) || (bytesMatch && hashMatch)) {
                                            it = targetLayerState.decodeCacheEntries.erase(it);
                                        } else {
                                            ++it;
                                        }
                                    }
                                    CPUAttention::H2OSharedState::LayerState::DecodedCacheEntry entry;
                                    entry.startToken = stored.startToken;
                                    entry.tokenCount = stored.tokenCount;
                                    entry.rawHash = stored.rawHash;
                                    entry.blobHash = stored.blobHash;
                                    entry.rawBytes = stored.rawBytes;
                                    entry.compressedBytes = stored.compressedBytes;
                                    entry.keyDecoded.swap(seededKey);
                                    entry.valueDecoded.swap(seededValue);
                                    targetLayerState.decodeCacheEntries.push_back(std::move(entry));
                                    while ((int)targetLayerState.decodeCacheEntries.size() > decodeCacheBlocks) {
                                        targetLayerState.decodeCacheEntries.pop_front();
                                    }
                                }
                            }
                        }
                        if (zeroKvRange(targetCacheManager,
                                        mKvNumHead,
                                        mHeadDim,
                                        mBytes,
                                        lP,
                                        hP,
                                        targetFlashBlockKv,
                                        stored.startToken,
                                        stored.tokenCount)) {
                            stored.rawDropped = true;
                        } else {
                            targetLayerState.losslessFallbackCount += 1;
                            MNN_ERROR("[H2O-LOSSLESS] zeroKvRange failed: target_layer=%d current_layer=%d start=%d tokens=%d kv_len=%d max_len=%d\n",
                                      targetLayerIndex,
                                      layerIndex,
                                      stored.startToken,
                                      stored.tokenCount,
                                      targetCacheManager->kvLength(),
                                      targetCacheManager->maxLength());
                        }
                    }
                }
                if (!runtimeStoreModeLocal) {
                    decodeOnePendingLosslessBlock(targetLayerState);
                }
                if (countRatioStats) {
                    targetLayerState.losslessRawBytes += stats.rawBytes;
                    targetLayerState.losslessCompressedBytes += stats.compressedBytes;
                }
                targetLayerState.losslessDecompressedBytes += stats.decompressedBytes;
                targetLayerState.losslessCompressUs += stats.compressUs;
                targetLayerState.losslessDecompressUs += stats.decompressUs;
                targetLayerState.losslessFallbackCount += stats.fallbackUsed ? 1 : 0;
                if (updateSchedule) {
                    targetLayerState.losslessUpdateCount += 1;
                    targetLayerState.losslessLastTokenBudget = (runtimeProbeMode && bootstrapWindowSampled)
                        ? coldTokenBudget
                        : (startToken + tokenCount);
                    targetLayerState.losslessLastStep = scheduleStep;
                }

                mH2OState->globalLastLosslessCodecUs = stats.compressUs;
                mH2OState->globalLastLosslessStep = scheduleStep;
                mH2OState->globalLastLosslessTokenBudget = targetLayerState.losslessLastTokenBudget;
                mH2OState->globalLastLosslessRawBytes = targetLayerState.losslessRawBytes;
                mH2OState->globalLastLosslessCompressedBytes = targetLayerState.losslessCompressedBytes;
                mH2OState->globalLastLosslessDecompressedBytes = targetLayerState.losslessDecompressedBytes;
                mH2OState->globalLastLosslessCompressUs = targetLayerState.losslessCompressUs;
                mH2OState->globalLastLosslessDecompressUs = targetLayerState.losslessDecompressUs;
                mH2OState->globalLosslessFallbackCount = targetLayerState.losslessFallbackCount;
                mH2OState->globalLosslessBackpressureSkipCount =
                    targetLayerState.losslessBackpressureSkipCount;
                if (targetLayerState.losslessCompressedBytes > 0 && targetLayerState.losslessRawBytes > 0) {
                    mH2OState->globalLastLosslessRatio =
                        (float)((double)targetLayerState.losslessRawBytes / (double)targetLayerState.losslessCompressedBytes);
                }
                const int64_t pendingAfterUpdate = (int64_t)targetLayerState.losslessBlocks.size()
                    + getAsyncPendingCount();
                mH2OState->globalLosslessQueueDepthPeak =
                    ALIMAX(mH2OState->globalLosslessQueueDepthPeak, pendingAfterUpdate);
                if (mMeta->h2o_log_stats != 0) {
                    const bool zstdReady = getZstdApi().available;
                    MNN_PRINT("[H2O-LOSSLESS] layer=%d kv=%d start=%d tokens=%d raw=%llu attempt_comp=%llu eff_comp=%llu attempt_ratio=%.4f eff_ratio=%.4f comp_us=%lld decomp_us=%lld updates=%lld fallback=%d backpressure=%d queue=%d async_q_peak=%lld async_wait_us=%lld mode=%d zstd=%d\n",
                              targetLayerIndex,
                              kvSeqLen,
                              startToken,
                              tokenCount,
                              (unsigned long long)stats.rawBytes,
                              (unsigned long long)stats.attemptedCompressedBytes,
                              (unsigned long long)stats.compressedBytes,
                              stats.attemptedRatio,
                              stats.ratio,
                              (long long)stats.compressUs,
                              (long long)targetLayerState.losslessDecompressUs,
                              (long long)targetLayerState.losslessUpdateCount,
                              stats.fallbackUsed ? 1 : 0,
                              stats.backpressureSkipped ? 1 : 0,
                              (int)pendingAfterUpdate,
                              (long long)mH2OState->globalLosslessAsyncQueuePeak,
                              (long long)mH2OState->globalLosslessAsyncWaitUs,
                              runtimeMode,
                              zstdReady ? 1 : 0);
                }
            };
        auto submitAsyncEncodeTask = [&](CPUAttention::H2OSharedState::AsyncTask&& task) -> bool {
            if (mH2OState == nullptr) {
                return false;
            }
            std::lock_guard<std::mutex> lock(mH2OState->asyncMutex);
            if (mH2OState->asyncWorkers.empty() || mH2OState->asyncConfiguredThreads <= 0) {
                return false;
            }
            mH2OState->asyncPendingTasks += 1;
            mH2OState->asyncTasks.emplace_back(std::move(task));
            mH2OState->globalLosslessAsyncQueuePeak =
                ALIMAX(mH2OState->globalLosslessAsyncQueuePeak, mH2OState->asyncPendingTasks);
            mH2OState->asyncCv.notify_one();
            return true;
        };
        auto drainAsyncResults = [&](bool waitAll) {
            if (mH2OState == nullptr) {
                return;
            }
            while (true) {
                std::deque<CPUAttention::H2OSharedState::AsyncResult> ready;
                bool allDone = false;
                {
                    std::unique_lock<std::mutex> lock(mH2OState->asyncMutex);
                    if (waitAll) {
                        if (mH2OState->asyncPendingTasks > 0 && mH2OState->asyncCompleted.empty()) {
                            auto w0 = std::chrono::high_resolution_clock::now();
                            mH2OState->asyncDoneCv.wait(lock, [&]() {
                                return mH2OState->asyncPendingTasks == 0 || !mH2OState->asyncCompleted.empty();
                            });
                            auto w1 = std::chrono::high_resolution_clock::now();
                            mH2OState->globalLosslessAsyncWaitUs +=
                                (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(w1 - w0).count();
                        }
                    } else if (mH2OState->asyncCompleted.empty()) {
                        return;
                    }
                    ready.swap(mH2OState->asyncCompleted);
                    allDone = (mH2OState->asyncPendingTasks == 0) && mH2OState->asyncCompleted.empty();
                }
                for (auto& asyncStats : ready) {
                    LosslessRuntimeStats stats;
                    stats.ratio = asyncStats.ratio;
                    stats.attemptedRatio = asyncStats.attemptedRatio;
                    stats.rawBytes = asyncStats.rawBytes;
                    stats.compressedBytes = asyncStats.compressedBytes;
                    stats.attemptedCompressedBytes = asyncStats.attemptedCompressedBytes;
                    stats.decompressedBytes = asyncStats.decompressedBytes;
                    stats.compressUs = asyncStats.compressUs;
                    stats.decompressUs = asyncStats.decompressUs;
                    stats.fallbackUsed = asyncStats.fallbackUsed;
                    stats.backpressureSkipped = asyncStats.backpressureSkipped;
                    stats.rawHash = asyncStats.rawHash;
                    stats.blobHash = asyncStats.blobHash;
                    stats.keyBlob.swap(asyncStats.keyBlob);
                    stats.valueBlob.swap(asyncStats.valueBlob);
                    applyEncodedStatsToLayer(
                        asyncStats.layerIndex,
                        asyncStats.startToken,
                        asyncStats.tokenCount,
                        std::move(stats),
                        false,
                        false,
                        0,
                        mH2OState->globalTokenStep);
                }
                if (!waitAll || allDone) {
                    break;
                }
            }
        };
        drainAsyncResults(false);
        if (layerIndex == layerCount - 1 && getAsyncPendingCount() > 0) {
            drainAsyncResults(true);
        }
        if (mMeta->h2o_lossless_enable != 0
            && mMeta->h2o_lossless_codec == 1
            && mMeta->h2o_lossless_runtime_enable != 0) {
            const bool inFrontLosslessRange = layerIndex < ALIMAX(0, mMeta->h2o_lossless_front_n);
            const bool inH2OKeptLosslessRange = h2oLayerInRange;
            if (mMeta->h2o_lossless_scope == 1) { // front_n
                applyLossless = inFrontLosslessRange;
                losslessTokenBudget = kvSeqLen;
            } else if (mMeta->h2o_lossless_scope == 2) { // h2o_kept
                applyLossless = inH2OKeptLosslessRange;
                losslessTokenBudget = finalKeepTokensForLayer;
            } else if (mMeta->h2o_lossless_scope == 3) { // front_n_and_h2o_kept
                applyLossless = inFrontLosslessRange || inH2OKeptLosslessRange;
                // Prefer full front-layer budget; for deeper H2O-kept layers use post-lossy keep budget.
                losslessTokenBudget = inFrontLosslessRange ? kvSeqLen : finalKeepTokensForLayer;
            }
            if (runtimeStoreModeLocal && mMeta->h2o_lossless_store_disable_front != 0 && inFrontLosslessRange) {
                if (mMeta->h2o_lossless_scope == 1) {
                    applyLossless = false;
                } else if (mMeta->h2o_lossless_scope == 3) {
                    applyLossless = inH2OKeptLosslessRange;
                    if (applyLossless) {
                        losslessTokenBudget = finalKeepTokensForLayer;
                    }
                }
            }
            if (applyLossless && inFrontLosslessRange && !runtimeProbeMode) {
                const int frontSampleIntervalRaw = mMeta->h2o_lossless_front_sample_token_interval;
                if (frontSampleIntervalRaw > 1) {
                    const int64_t tokenStep = std::max<int64_t>(1, mH2OState->globalTokenStep);
                    const bool tokenSampled = ((tokenStep - 1) % (int64_t)frontSampleIntervalRaw) == 0;
                    applyLossless = tokenSampled;
                }
            }
        }
        if (applyLossless && runtimeMode == 1
            && (mMeta->h2o_lossless_scope == 2 || mMeta->h2o_lossless_scope == 3)) {
            const bool inFrontLosslessRange = layerIndex < ALIMAX(0, mMeta->h2o_lossless_front_n);
            const bool inH2OKeptLosslessRange = h2oLayerInRange;
            if (inH2OKeptLosslessRange && !(mMeta->h2o_lossless_scope == 3 && inFrontLosslessRange)) {
                const int keptSampleLayers = mMeta->h2o_lossless_kept_sample_layers;
                const int keptSampleTokenInterval = mMeta->h2o_lossless_kept_sample_token_interval;
                const int keptLayerOffset = layerIndex - h2oLayerStart;
                const bool layerSampled = (keptSampleLayers <= 0)
                    ? (keptLayerOffset >= 0)
                    : (keptLayerOffset >= 0 && keptLayerOffset < keptSampleLayers);
                const int64_t tokenStep = std::max<int64_t>(1, mH2OState->globalTokenStep);
                const bool tokenSampled = (keptSampleTokenInterval <= 1)
                    ? true
                    : (((tokenStep - 1) % (int64_t)keptSampleTokenInterval) == 0);
                // Full-mode deep kept range can be tuned by kept-layer count and decode-step interval.
                // This preserves front-layer signal while capping decode overhead in deep layers.
                applyLossless = layerSampled && tokenSampled;
                if (applyLossless) {
                    losslessTokenBudget = finalKeepTokensForLayer;
                }
            }
        }
        if (applyLossless && mMeta->h2o_lossless_runtime_enable != 0 && runtimeProbeMode) {
            // Runtime path is a compression-statistics probe, not a storage rewrite.
            // Sample one representative layer per scope to cap decode perturbation.
            if (mMeta->h2o_lossless_scope == 1) { // front_n
                applyLossless = (layerIndex == 0);
            } else if (mMeta->h2o_lossless_scope == 2) { // h2o_kept
                applyLossless = (layerIndex == h2oLayerStart);
            } else if (mMeta->h2o_lossless_scope == 3) { // front_n_and_h2o_kept
                const bool sampleFront = (layerIndex == 0);
                const bool sampleKept = (layerIndex == h2oLayerStart);
                applyLossless = sampleFront || sampleKept;
            }
        }
        if (applyLossless && layerIndex >= 0 && layerIndex < (int)mH2OState->layerStates.size()) {
            auto& layerState = mH2OState->layerStates[layerIndex];
            auto countPendingBlocks = [&](const CPUAttention::H2OSharedState::LayerState& state) -> int64_t {
                int64_t pending = 0;
                for (const auto& block : state.losslessBlocks) {
                    const bool pendingBlock = runtimeStoreModeLocal ? block.rawDropped : !block.decodedOnce;
                    if (pendingBlock) {
                        pending += 1;
                    }
                }
                return pending;
            };
            const int hotSinkTokens = ALIMAX(0, mMeta->h2o_lossless_hot_sink_tokens);
            const int hotRecentTokens = ALIMAX(0, mMeta->h2o_lossless_hot_recent_tokens);
            const int coldBeginToken = ALIMIN(losslessTokenBudget, hotSinkTokens);
            const int coldEndToken = ALIMAX(coldBeginToken, losslessTokenBudget - hotRecentTokens);
            const int coldTokenBudget = coldEndToken;
            if (coldTokenBudget < layerState.losslessLastTokenBudget) {
                // Shrink can invalidate previously staged cold blocks.
                // In full mode, decode as many pending blocks as possible before
                // clear so fallback reflects real decode failures, not queue residue.
                if (!runtimeStoreModeLocal) {
                    const int64_t pendingBefore = countPendingBlocks(layerState);
                    int64_t attempts = 0;
                    while (attempts < pendingBefore && countPendingBlocks(layerState) > 0) {
                        decodeOnePendingLosslessBlock(layerState);
                        attempts += 1;
                    }
                }
                // Count remaining pending blocks that will be discarded by clear().
                const int64_t discardedUndecoded = countPendingBlocks(layerState);
                if (discardedUndecoded > 0) {
                    if (!runtimeStoreModeLocal) {
                        layerState.losslessFallbackCount += discardedUndecoded;
                        MNN_PRINT("[H2O-LOSSLESS] shrink: discarding %lld undecoded blocks\n", (long long)discardedUndecoded);
                    } else {
                        // Store mode uses dropped-raw compressed blocks as steady state.
                        // Shrink cleanup is expected and should not be treated as fallback.
                        MNN_PRINT("[H2O-LOSSLESS] shrink(store): discarding %lld staged blocks\n", (long long)discardedUndecoded);
                    }
                }
                // Compressible cold budget shrank (eviction or hot-window shift).
                // Reset cold-window watermark so delta tracking works going forward,
                // while preserving accumulated byte/ratio stats for aggregation.
                layerState.losslessLastStep = 0;
                layerState.losslessLastTokenBudget = coldTokenBudget;
                layerState.pendingRemove = 0;
                layerState.pendingPlanReady = 0;
                layerState.activeReservePairs.clear();
                layerState.pendingReservePairs.clear();
                layerState.losslessBlocks.clear();
                layerState.decodeCacheEntries.clear();
            }

            const int64_t pendingQueueDepth = countPendingBlocks(layerState) + getAsyncPendingCount();
            mH2OState->globalLosslessQueueDepthPeak =
                ALIMAX(mH2OState->globalLosslessQueueDepthPeak, pendingQueueDepth);
            mH2OState->globalLosslessAsyncQueuePeak =
                ALIMAX(mH2OState->globalLosslessAsyncQueuePeak, getAsyncPendingCount());

            const int triggerMin = ALIMAX(1, mMeta->h2o_trigger_min_tokens);
            const int updateInterval = ALIMAX(1, mMeta->h2o_update_interval);
            const int blockStep = ALIMAX(1, mMeta->h2o_lossless_block_tokens);
            const int storeGroupedStepCfg = ALIMAX(0, mMeta->h2o_lossless_store_grouped_step_tokens);
            const int groupedStep = runtimeStoreModeLocal
                ? ALIMAX(1, storeGroupedStepCfg > 0 ? storeGroupedStepCfg : (blockStep * 3))
                : blockStep;
            const int storeOverlapTokens = runtimeStoreModeLocal ? ALIMAX(0, groupedStep / 8) : 0;
            const int64_t losslessStep = mH2OState->globalTokenStep;
            const int tokenBudgetGrowth = coldTokenBudget - layerState.losslessLastTokenBudget;
            const bool intervalReady = (losslessStep - layerState.losslessLastStep >= updateInterval);
            const int storeBootstrapCfg = ALIMAX(0, mMeta->h2o_lossless_store_bootstrap_tokens);
            const int storeBootstrapTokens = ALIMAX(1, storeBootstrapCfg > 0 ? storeBootstrapCfg : ALIMAX(16, blockStep / 2));
            const int runtimeBootstrapSampleCap = runtimeProbeMode
                ? 32
                : (runtimeStoreModeLocal ? storeBootstrapTokens : blockStep);
            const int bootstrapSampleBase = runtimeStoreModeLocal ? storeBootstrapTokens : blockStep;
            const int bootstrapSampleTokens = ALIMAX(1, ALIMIN(bootstrapSampleBase, runtimeBootstrapSampleCap));

            int evalTokenCount = 0;
            int evalStartToken = ALIMAX(layerState.losslessLastTokenBudget, coldBeginToken);
            bool bootstrapWindowSampled = false;
            if (kvSeqLen >= triggerMin && coldEndToken > coldBeginToken) {
                if (layerState.losslessUpdateCount <= 0) {
                    // Probe mode uses a tiny prefix sample; full mode keeps full block.
                    if (tokenBudgetGrowth >= bootstrapSampleTokens) {
                        evalTokenCount = ALIMIN(bootstrapSampleTokens, coldEndToken - coldBeginToken);
                        evalStartToken = runtimeProbeMode
                            ? coldBeginToken
                            : ALIMAX(coldBeginToken, coldEndToken - bootstrapSampleTokens);
                        bootstrapWindowSampled = true;
                    }
                } else if (intervalReady && (tokenBudgetGrowth + storeOverlapTokens) >= groupedStep) {
                    const int startNoOverlap = ALIMAX(layerState.losslessLastTokenBudget, coldBeginToken);
                    evalStartToken = runtimeStoreModeLocal
                        ? ALIMAX(startNoOverlap - storeOverlapTokens, coldBeginToken)
                        : startNoOverlap;
                    evalTokenCount = ALIMIN(groupedStep, ALIMAX(0, coldEndToken - evalStartToken));
                }
            }

            if (evalTokenCount > 0) {
                const int maxQueue = ALIMAX(1, mMeta->h2o_lossless_max_queue);
                const int64_t pending = countPendingBlocks(layerState) + getAsyncPendingCount();
                if (pending >= maxQueue) {
                    LosslessRuntimeStats stats;
                    // Backpressure skip is a scheduling throttle, not a codec fallback failure.
                    stats.fallbackUsed = false;
                    stats.backpressureSkipped = true;
                    layerState.losslessBackpressureSkipCount += 1;
                    applyEncodedStatsToLayer(
                        layerIndex,
                        evalStartToken,
                        evalTokenCount,
                        std::move(stats),
                        true,
                        bootstrapWindowSampled,
                        coldTokenBudget,
                        losslessStep);
                } else if (asyncEncodeEnabled) {
                    drainAsyncResults(false);
                    auto payload = collectLosslessRuntimePayload(evalStartToken, evalTokenCount);
                    if (payload.fallbackUsed || payload.rawBytes == 0) {
                        LosslessRuntimeStats stats;
                        // rawBytes==0 is a benign no-op sample; only propagate real fallback flags.
                        stats.fallbackUsed = payload.fallbackUsed;
                        applyEncodedStatsToLayer(
                            layerIndex,
                            evalStartToken,
                            evalTokenCount,
                            std::move(stats),
                            true,
                            bootstrapWindowSampled,
                            coldTokenBudget,
                            losslessStep);
                    } else {
                        layerState.losslessUpdateCount += 1;
                        layerState.losslessLastTokenBudget = (runtimeProbeMode && bootstrapWindowSampled)
                            ? coldTokenBudget
                            : (evalStartToken + evalTokenCount);
                        layerState.losslessLastStep = losslessStep;

	                        CPUAttention::H2OSharedState::AsyncTask task;
	                        {
	                            std::lock_guard<std::mutex> lock(mH2OState->asyncMutex);
	                            task.taskId = ++mH2OState->asyncTaskSerial;
	                        }
	                        task.layerIndex = layerIndex;
	                        task.startToken = evalStartToken;
	                        task.tokenCount = evalTokenCount;
	                        task.runtimeStoreMode = runtimeStoreModeLocal;
	                        auto payloadForAsync = std::make_shared<LosslessCollectPayload>(std::move(payload));
	                        const int bytes = mBytes;
                        task.fn = [payloadForAsync, layerIndex, runtimeStoreModeLocal, bytes, keyFlags, valueFlags]() mutable {
                            auto& payload = *payloadForAsync;
                            CPUAttention::H2OSharedState::AsyncResult asyncStats;
                            asyncStats.layerIndex = layerIndex;
                            asyncStats.startToken = payload.evalStart;
                            asyncStats.tokenCount = payload.evalTokens;
                            asyncStats.runtimeStoreMode = runtimeStoreModeLocal;
                            asyncStats.rawBytes = payload.rawBytes;
                            asyncStats.rawHash = payload.rawHash;
                            if (payload.fallbackUsed || payload.rawBytes == 0) {
                                // rawBytes==0 is a benign no-op payload; only keep real fallback flags.
                                asyncStats.fallbackUsed = payload.fallbackUsed;
                                return asyncStats;
                            }
                            auto c0 = std::chrono::high_resolution_clock::now();
                            bool keyOk = false;
                            bool valueOk = false;
                            if (!payload.keyMerged.empty()) {
                                if (bytes == 2) {
                                    keyOk = encodeFp16GearPredictive(payload.keyMerged.data(), payload.keyMerged.size(), keyFlags, keyFlags, asyncStats.keyBlob);
                                } else {
                                    keyOk = encodeFp32LanePredictive(payload.keyMerged.data(), payload.keyMerged.size(), keyFlags, asyncStats.keyBlob);
                                }
                            } else {
                                keyOk = true;
                            }
                            if (!payload.valueMerged.empty()) {
                                if (bytes == 2) {
                                    valueOk = encodeFp16GearPredictive(payload.valueMerged.data(), payload.valueMerged.size(), valueFlags, valueFlags, asyncStats.valueBlob);
                                } else {
                                    valueOk = encodeFp32LanePredictive(payload.valueMerged.data(), payload.valueMerged.size(), valueFlags, asyncStats.valueBlob);
                                }
                            } else {
                                valueOk = true;
                            }
                            auto c1 = std::chrono::high_resolution_clock::now();
                            asyncStats.compressUs = (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(c1 - c0).count();
                            if (!keyOk || !valueOk) {
                                asyncStats.fallbackUsed = true;
                                asyncStats.compressedBytes = asyncStats.rawBytes;
                                asyncStats.attemptedCompressedBytes = asyncStats.rawBytes;
                                asyncStats.attemptedRatio = 1.0f;
                                asyncStats.ratio = 1.0f;
                                asyncStats.keyBlob.clear();
                                asyncStats.valueBlob.clear();
                                return asyncStats;
                            }
                            const uint64_t attemptedBytes = (uint64_t)asyncStats.keyBlob.size() + (uint64_t)asyncStats.valueBlob.size();
                            if (attemptedBytes == 0) {
                                asyncStats.fallbackUsed = true;
                                asyncStats.compressedBytes = asyncStats.rawBytes;
                                asyncStats.attemptedCompressedBytes = asyncStats.rawBytes;
                                asyncStats.attemptedRatio = 1.0f;
                                asyncStats.ratio = 1.0f;
                                asyncStats.keyBlob.clear();
                                asyncStats.valueBlob.clear();
                                return asyncStats;
                            }
                            if (attemptedBytes >= asyncStats.rawBytes) {
                                // No-gain compression path: keep raw bytes without counting fallback.
                                asyncStats.fallbackUsed = false;
                                asyncStats.compressedBytes = asyncStats.rawBytes;
                                asyncStats.attemptedCompressedBytes = attemptedBytes;
                                asyncStats.attemptedRatio =
                                    static_cast<float>((double)asyncStats.rawBytes / (double)attemptedBytes);
                                asyncStats.ratio = 1.0f;
                                asyncStats.keyBlob.clear();
                                asyncStats.valueBlob.clear();
                                return asyncStats;
                            }
                            asyncStats.attemptedCompressedBytes = attemptedBytes;
                            asyncStats.compressedBytes = attemptedBytes;
                            asyncStats.attemptedRatio = static_cast<float>((double)asyncStats.rawBytes / (double)attemptedBytes);
                            asyncStats.ratio = asyncStats.attemptedRatio;
                            asyncStats.blobHash = fnv1a64(asyncStats.valueBlob.data(), asyncStats.valueBlob.size(),
                                                          fnv1a64(asyncStats.keyBlob.data(), asyncStats.keyBlob.size()));
                            return asyncStats;
                        };
                        if (!submitAsyncEncodeTask(std::move(task))) {
                            auto stats = encodeLosslessRuntimePayload(*payloadForAsync);
                            applyEncodedStatsToLayer(
                                layerIndex,
                                evalStartToken,
                                evalTokenCount,
                                std::move(stats),
                                false,
                                false,
                                coldTokenBudget,
                                losslessStep);
                        }
                    }
                } else {
                    auto stats = encodeLosslessRuntimeStats(evalStartToken, evalTokenCount);
                    if (!stats.fallbackUsed) {
                        const int64_t pendingNow = countPendingBlocks(layerState);
                        if (pendingNow >= maxQueue) {
                            // Sync path backpressure: keep as throttle signal, not fallback failure.
                            stats.fallbackUsed = false;
                            stats.backpressureSkipped = true;
                            stats.compressedBytes = stats.rawBytes;
                            stats.ratio = 1.0f;
                            stats.keyBlob.clear();
                            stats.valueBlob.clear();
                            layerState.losslessBackpressureSkipCount += 1;
                        }
                    }
                    applyEncodedStatsToLayer(
                        layerIndex,
                        evalStartToken,
                        evalTokenCount,
                        std::move(stats),
                        true,
                        bootstrapWindowSampled,
                        coldTokenBudget,
                        losslessStep);
                }
            }
        }

        drainAsyncResults(false);
        if (layerIndex == layerCount - 1 && getAsyncPendingCount() > 0) {
            drainAsyncResults(true);
        }

        uint64_t totalRawBytes = 0;
        uint64_t totalCompressedBytes = 0;
        uint64_t totalDecompressedBytes = 0;
        int64_t totalCompressUs = 0;
        int64_t totalDecompressUs = 0;
        int64_t totalFallbackCount = 0;
        int64_t totalBackpressureSkipCount = 0;
        const int losslessScope = mMeta->h2o_lossless_scope;
        const int frontN = ALIMAX(0, mMeta->h2o_lossless_front_n);
        bool hasIncludedLayer = false;
        for (int i = 0; i < (int)mH2OState->layerStates.size(); ++i) {
            bool include = false;
            if (losslessScope == 1) { // front_n
                include = (i < frontN);
            } else if (losslessScope == 2) { // h2o_kept range
                include = (i >= h2oLayerStart && i <= h2oLayerEnd);
            } else if (losslessScope == 3) { // front_n_and_h2o_kept
                include = (i < frontN) || (i >= h2oLayerStart && i <= h2oLayerEnd);
            }
            if (!include) {
                continue;
            }
            const auto& layerState = mH2OState->layerStates[i];
            if (layerState.losslessUpdateCount <= 0) {
                continue;
            }
            hasIncludedLayer = true;
            totalRawBytes += layerState.losslessRawBytes;
            totalCompressedBytes += layerState.losslessCompressedBytes;
            totalDecompressedBytes += layerState.losslessDecompressedBytes;
            totalCompressUs += layerState.losslessCompressUs;
            totalDecompressUs += layerState.losslessDecompressUs;
            totalFallbackCount += layerState.losslessFallbackCount;
            totalBackpressureSkipCount += layerState.losslessBackpressureSkipCount;
        }

        // Safety fallback: if scope-filtered selection yields no valid layer,
        // aggregate all layers that have runtime lossless updates.
        if (!hasIncludedLayer) {
            totalRawBytes = 0;
            totalCompressedBytes = 0;
            totalDecompressedBytes = 0;
            totalCompressUs = 0;
            totalDecompressUs = 0;
            totalFallbackCount = 0;
            totalBackpressureSkipCount = 0;
            for (int i = 0; i < (int)mH2OState->layerStates.size(); ++i) {
                const auto& layerState = mH2OState->layerStates[i];
                if (layerState.losslessUpdateCount <= 0) {
                    continue;
                }
                totalRawBytes += layerState.losslessRawBytes;
                totalCompressedBytes += layerState.losslessCompressedBytes;
                totalDecompressedBytes += layerState.losslessDecompressedBytes;
                totalCompressUs += layerState.losslessCompressUs;
                totalDecompressUs += layerState.losslessDecompressUs;
                totalFallbackCount += layerState.losslessFallbackCount;
                totalBackpressureSkipCount += layerState.losslessBackpressureSkipCount;
            }
        }

        mH2OState->globalLosslessBackpressureSkipCount = totalBackpressureSkipCount;
        if (totalRawBytes > 0 && totalCompressedBytes > 0) {
            mH2OState->globalLastLosslessRawBytes = totalRawBytes;
            mH2OState->globalLastLosslessCompressedBytes = totalCompressedBytes;
            mH2OState->globalLastLosslessDecompressedBytes = totalDecompressedBytes;
            mH2OState->globalLastLosslessCompressUs = totalCompressUs;
            mH2OState->globalLastLosslessDecompressUs = totalDecompressUs;
            mH2OState->globalLosslessFallbackCount = totalFallbackCount;
        }
        if (totalCompressedBytes > 0 && totalRawBytes > 0) {
            mH2OState->globalLastLosslessRatio =
                (float)((double)totalRawBytes / (double)totalCompressedBytes);
        }

        // Only overwrite meta stats when we have valid lossless byte accounting.
        // This prevents later non-lossless layers from clobbering earlier valid stats
        // with per-layer default values (1.0 / 0.0) in the bench summary path.
        const bool hasValidLosslessStats =
            mH2OState->globalLastLosslessRawBytes > 0
            && mH2OState->globalLastLosslessCompressedBytes > 0;
        if (hasValidLosslessStats) {
            mMeta->h2o_lossless_ratio = mH2OState->globalLastLosslessRatio;
            mMeta->h2o_codec_us = mH2OState->globalLastLosslessCodecUs;
            mMeta->h2o_lossless_raw_bytes = mH2OState->globalLastLosslessRawBytes;
            mMeta->h2o_lossless_compressed_bytes = mH2OState->globalLastLosslessCompressedBytes;
            mMeta->h2o_lossless_decompressed_bytes = mH2OState->globalLastLosslessDecompressedBytes;
            mMeta->h2o_lossless_compress_us = mH2OState->globalLastLosslessCompressUs;
            mMeta->h2o_lossless_decompress_us = mH2OState->globalLastLosslessDecompressUs;
            mMeta->h2o_lossless_fallback_count = mH2OState->globalLosslessFallbackCount;
        }
        mMeta->h2o_lossless_queue_depth_peak = mH2OState->globalLosslessQueueDepthPeak;
        mMeta->h2o_lossless_async_queue_peak = mH2OState->globalLosslessAsyncQueuePeak;
        mMeta->h2o_lossless_async_wait_us = mH2OState->globalLosslessAsyncWaitUs;
        mMeta->h2o_lossless_decode_cache_hit = mH2OState->globalLosslessDecodeCacheHit;
        mMeta->h2o_lossless_decode_cache_miss = mH2OState->globalLosslessDecodeCacheMiss;
        mMeta->h2o_lossless_backpressure_skip_count = mH2OState->globalLosslessBackpressureSkipCount;
    }

    backend()->onReleaseBuffer(unpackQK.get(), Backend::STATIC);
    backend()->onReleaseBuffer(softmMaxQ.get(), Backend::STATIC);
    backend()->onReleaseBuffer(newPackQK.get(), Backend::STATIC);
    backend()->onReleaseBuffer(mTempQKBlock.get(), Backend::STATIC);

    if (!mKVCache) {
        mKVCacheManager->onClear();
    }
    auto ptr = outputs[0]->host<float>();
    if (seqLen < outputs[0]->length(1)) {
        ::memset(outputs[0]->host<uint8_t>() + seqLen * mHeadDim * mNumHead * mBytes, 0, (outputs[0]->length(1)-seqLen) * mHeadDim * mNumHead * mBytes);
    }
    return NO_ERROR;
}

bool CPUAttention::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto tmp = new CPUAttention(bn, mKVCache);
    tmp->mKVCacheManager = mKVCacheManager;
    tmp->mH2OState = mH2OState;
    *dst = tmp;
    return true;
}

CPUAttention::CPUAttention(Backend *backend, bool kv_cache) : Execution(backend), mKVCache(kv_cache) {
    mMeta = (KVMeta*)(backend->getMetaPtr());
    mPackQ.reset(Tensor::createDevice<float>({1, 1, 1, 1}));
    mPackQKV.reset(Tensor::createDevice<float>({1, 1, 1, 1}));
    MNN::KVCacheManager::KVCacheConfig kvconfig;

    // attentionOption % 8:
    // 0: Do not quantize
    // 1: Q,K: Int8, V: Float32
    // 2: Q,K,V: Int8

    // attentionOption / 8:
    // 0: do not use flash attention
    // 1: use flash attention
    kvconfig.mKVCacheDir = static_cast<CPUBackend *>(backend)->getRuntime()->hint().kvcacheDirPath;
    kvconfig.mPrefixCacheDir = static_cast<CPUBackend *>(backend)->getRuntime()->hint().prefixcacheDirPath;
    kvconfig.mExpandChunk = 64;
    kvconfig.mBlockNum = 1;
    mKVCacheManager.reset(new CPUKVCacheManager(backend, kvconfig));
    {
        // Share one H2O runtime state across all attention layers that bind to the same KVMeta.
        // Otherwise each layer keeps isolated cursors/statistics and the final meta can be overwritten
        // by a layer-local default state (showing 1.0/0.0 in bench summary).
        static std::mutex sH2OStateLock;
        static std::unordered_map<const void*, std::weak_ptr<H2OSharedState>> sH2OStateByMeta;
        const void* key = static_cast<const void*>(mMeta);
        std::lock_guard<std::mutex> guard(sH2OStateLock);
        auto it = sH2OStateByMeta.find(key);
        if (it != sH2OStateByMeta.end()) {
            mH2OState = it->second.lock();
            if (!mH2OState) {
                // Stale entry — weak_ptr expired; remove before reinserting.
                sH2OStateByMeta.erase(it);
            }
        }
        if (!mH2OState) {
            mH2OState.reset(new H2OSharedState);
            sH2OStateByMeta[key] = mH2OState;
        }
    }
}

CPUAttention::~CPUAttention() {
    if (mH2OState) {
        std::vector<std::thread> workers;
        {
            std::lock_guard<std::mutex> lock(mH2OState->asyncMutex);
            // Worker lambdas capture shared state by value, so use_count cannot
            // be used as a shutdown gate. Always request stop and join here.
            // Drop queued tasks on shutdown; let in-flight workers finish naturally.
            if (!mH2OState->asyncTasks.empty()) {
                mH2OState->asyncTasks.clear();
                mH2OState->asyncPendingTasks = mH2OState->asyncRunningTasks;
            }
            mH2OState->asyncStop = true;
            mH2OState->asyncCv.notify_all();
            workers.swap(mH2OState->asyncWorkers);
        }
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        {
            std::lock_guard<std::mutex> lock(mH2OState->asyncMutex);
            mH2OState->asyncStop = false;
            mH2OState->asyncConfiguredThreads = 0;
            mH2OState->asyncPendingTasks = 0;
            mH2OState->asyncRunningTasks = 0;
            mH2OState->asyncTasks.clear();
            mH2OState->asyncCompleted.clear();
        }
    }
}

class CPUAttentionCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_AttentionParam();
        return new CPUAttention(backend, param->kv_cache());
    }
};

REGISTER_CPU_OP_CREATOR_TRANSFORMER(CPUAttentionCreator, OpType_Attention);

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
