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

struct LosslessStatSample {
    uint64_t rawBytes = 0;
    uint64_t compressedBytes = 0;
};

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
        api.isError = reinterpret_cast<unsigned(*)(size_t)>(dlsym(api.handle, "ZSTD_isError"));
        api.available = (api.compressBound != nullptr && api.compress != nullptr && api.isError != nullptr);
    }
#endif
    return api;
}

static size_t zstdEncodedSize(const uint8_t* data, size_t n, int level) {
    if (data == nullptr || n == 0) {
        return 0;
    }
    auto& api = getZstdApi();
    if (!api.available) {
        return 0;
    }
#if !defined(_WIN32)
    const size_t bound = api.compressBound(n);
    if (bound == 0 || bound < n / 2) {
        return 0;
    }
    std::vector<uint8_t> out(bound);
    const size_t ret = api.compress(out.data(), out.size(), data, n, level);
    if (api.isError(ret) != 0) {
        return 0;
    }
    return ret;
#else
    (void)level;
    return 0;
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
static size_t rleEncodedSize(const uint8_t* data, size_t n) {
    if (data == nullptr || n == 0) {
        return 0;
    }
    size_t outBytes = 0;
    size_t i = 0;
    while (i < n) {
        size_t run = repeatRunLen(data, n, i);
        if (run >= 4) {
            outBytes += 2;
            i += run;
            continue;
        }
        const size_t litBegin = i;
        size_t litLen = 0;
        while (i < n && litLen < 128) {
            run = repeatRunLen(data, n, i);
            if (run >= 4) {
                break;
            }
            ++i;
            ++litLen;
        }
        MNN_ASSERT(litLen > 0);
        outBytes += 1 + (i - litBegin);
    }
    return outBytes;
}

// Encode one byte stream with predictor search and payload compression.
// Frame bytes:
// - mode: 1 byte (0 raw, 1 delta_seq, 2 xor_seq)
// - codec: 1 byte (0 RLE, 1 zstd)
// - original length: 4 bytes (LE)
// - payload bytes
static size_t predictiveRLEEncodedSize(const std::vector<uint8_t>& src,
                                       const PredictorFlags& flags,
                                       std::vector<uint8_t>& temp) {
    if (src.empty()) {
        return 6;
    }
    size_t bestPayload = std::numeric_limits<size_t>::max();
    const int zstdLevel = 3;
    auto betterPayload = [&](const std::vector<uint8_t>& stream) -> size_t {
        const size_t rleBytes = rleEncodedSize(stream.data(), stream.size());
        size_t best = rleBytes;
        const size_t zstdBytes = zstdEncodedSize(stream.data(), stream.size(), zstdLevel);
        if (zstdBytes > 0) {
            best = ALIMIN(best, zstdBytes);
        }
        return best;
    };
    if (flags.raw) {
        bestPayload = ALIMIN(bestPayload, betterPayload(src));
    }
    if (flags.deltaSeq) {
        buildDeltaSeq(src, temp);
        bestPayload = ALIMIN(bestPayload, betterPayload(temp));
    }
    if (flags.xorSeq) {
        buildXorSeq(src, temp);
        bestPayload = ALIMIN(bestPayload, betterPayload(temp));
    }
    if (bestPayload == std::numeric_limits<size_t>::max()) {
        bestPayload = betterPayload(src);
    }
    // stream frame bytes:
    // mode(1) + codec(1) + raw_len(4) + payload
    return 1 + 1 + 4 + bestPayload;
}

// Runtime bitstream size for fp16 gear-split + predictor + payload compression
// (prefer zstd level-3 if available, fallback to RLE).
static size_t fp16GearPredictiveEncodedSize(const uint8_t* data,
                                            size_t bytes,
                                            const PredictorFlags& loFlags,
                                            const PredictorFlags& hiFlags) {
    if (data == nullptr || bytes < 2) {
        return bytes;
    }
    const size_t words = bytes / 2;
    std::vector<uint8_t> lo(words);
    std::vector<uint8_t> hi(words);
    for (size_t i = 0; i < words; ++i) {
        lo[i] = data[2 * i + 0];
        hi[i] = data[2 * i + 1];
    }
    std::vector<uint8_t> temp;
    // Tensor frame:
    // - word_count: 4 bytes
    // - lo stream frame
    // - hi stream frame
    size_t encoded = 4;
    encoded += predictiveRLEEncodedSize(lo, loFlags, temp);
    encoded += predictiveRLEEncodedSize(hi, hiFlags, temp);
    return encoded;
}

static LosslessStatSample encodeGearDeltaLosslessStats(const uint8_t* keyPtr,
                                                       size_t keyBytes,
                                                       const uint8_t* valuePtr,
                                                       size_t valueBytes,
                                                       const PredictorFlags& keyFlags,
                                                       const PredictorFlags& valueFlags) {
    LosslessStatSample sample;
    if (keyPtr != nullptr && keyBytes >= 2) {
        const size_t keyAlignedBytes = (keyBytes / 2) * 2;
        sample.rawBytes += keyAlignedBytes;
        sample.compressedBytes += fp16GearPredictiveEncodedSize(keyPtr, keyAlignedBytes, keyFlags, keyFlags);
    }
    if (valuePtr != nullptr && valueBytes >= 2) {
        const size_t valueAlignedBytes = (valueBytes / 2) * 2;
        sample.rawBytes += valueAlignedBytes;
        sample.compressedBytes += fp16GearPredictiveEncodedSize(valuePtr, valueAlignedBytes, valueFlags, valueFlags);
    }
    return sample;
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

    if (mKVCache && mMeta != nullptr) {
        // Apply previous-step H2O compact plan before touching KV cache in this step.
        if (mMeta->h2o_pending_plan_ready != 0
            && mMeta->remove == 0
            && mMeta->n_reserve == 0
            && mMeta->previous >= mMeta->h2o_pending_remove) {
            mMeta->remove = mMeta->h2o_pending_remove;
            mMeta->reserve = mMeta->h2o_pending_reserve;
            mMeta->n_reserve = mMeta->h2o_pending_n_reserve;
            mMeta->h2o_pending_plan_ready = 0;
            mMeta->h2o_pending_remove = 0;
            mMeta->h2o_pending_reserve = nullptr;
            mMeta->h2o_pending_n_reserve = 0;
        }
        if (mMeta->previous == mMeta->remove && mMeta->n_reserve == 0) {
            mKVCacheManager->onClear();
            mKVCacheManager->onAlloc(mMeta, seqLen);
        } else {
            MNN_ASSERT(mMeta->previous == mKVCacheManager->kvLength());
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

    const int layerCount = (mMeta != nullptr) ? ALIMAX(1, mMeta->layer_nums) : 1;
    int layerIndex = 0;
    if (mMeta != nullptr && mMeta->h2o_in_decode != 0 && mH2OState != nullptr) {
        layerIndex = static_cast<int>(mH2OState->decodeLayerCursor % static_cast<int64_t>(layerCount));
        mH2OState->decodeLayerCursor += 1;
        mH2OState->globalStep += 1;
    } else if (mMeta != nullptr && mH2OState != nullptr) {
        // New prefill/request path: clear decode-side lossless running stats.
        mH2OState->decodeLayerCursor = 0;
        mH2OState->globalStep = 0;
        mH2OState->globalLastTriggerStep = 0;
        mH2OState->globalLastLosslessStep = 0;
        mH2OState->globalLastLosslessRatio = 1.0f;
        mH2OState->globalLastLosslessCodecUs = 0;
        mH2OState->globalLastLosslessRawBytes = 0;
        mH2OState->globalLastLosslessCompressedBytes = 0;
        mH2OState->globalLastLosslessDecompressedBytes = 0;
        mH2OState->globalLastLosslessCompressUs = 0;
        mH2OState->globalLastLosslessDecompressUs = 0;
        mH2OState->globalLosslessQueueDepthPeak = 0;
        mH2OState->globalLosslessFallbackCount = 0;
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
        if ((int)mH2OState->layerStates.size() != 1) {
            mH2OState->layerStates.clear();
            mH2OState->layerStates.resize(1);
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
            targetKeep = ALIMAX(targetKeep, keptTokens);
            mMeta->h2o_target_keep_effective = kvSeqLen > 0 ? (float)targetKeep / (float)kvSeqLen : 1.0f;

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
            const int evictedTokens = ALIMAX(0, kvSeqLen - finalKeepTokens);
            const size_t reserveMetaBytes = (size_t)reservePairs.size() * sizeof(int);

            if (evictedTokens > 0 && !reservePairs.empty()) {
                mH2OState->reserveStorage.swap(reservePairs);
                mMeta->h2o_pending_remove = kvSeqLen;
                mMeta->h2o_pending_reserve = mH2OState->reserveStorage.data();
                mMeta->h2o_pending_n_reserve = (int)mH2OState->reserveStorage.size() / 2;
                mMeta->h2o_pending_plan_ready = 1;
                mMeta->h2o_total_evict_tokens += evictedTokens;
            }

            const size_t bytesPerToken = (size_t)mKvNumHead * (size_t)mHeadDim * (size_t)mBytes * 2;
            const size_t rawBefore = (size_t)kvSeqLen * bytesPerToken;
            const size_t rawAfter = (size_t)finalKeepTokens * bytesPerToken;
            const size_t lossyDen = rawAfter + reserveMetaBytes;
            mMeta->h2o_keep_ratio = kvSeqLen > 0 ? (float)finalKeepTokens / (float)kvSeqLen : 1.0f;
            mMeta->h2o_lossy_ratio = lossyDen > 0 ? (float)rawBefore / (float)lossyDen : 1.0f;
            mMeta->h2o_last_evict_tokens = evictedTokens;

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
            uint64_t rawBytes = 0;
            uint64_t compressedBytes = 0;
            uint64_t decompressedBytes = 0;
            bool fallbackUsed = false;
        };
        auto estimateLosslessRuntimeStats = [&](int tokenBudget) -> LosslessRuntimeStats {
            LosslessRuntimeStats stats;
            if (mMeta->h2o_lossless_codec != 1
                || mMeta->h2o_lossless_runtime_enable == 0
                || tokenBudget <= 0
                || kvSeqLen <= 0) {
                return stats;
            }
            if (mBytes != 2 && mBytes != 4) {
                stats.fallbackUsed = true;
                return stats;
            }
            const int evalTokens = clampInt(tokenBudget, 1, kvSeqLen);
            const double logicalRawBytes = (double)evalTokens * (double)mKvNumHead * (double)mHeadDim * (double)mBytes * 2.0;
            if (mBytes == 4) {
                // Runtime v3 lossless codec is currently fp16-only.
                stats.ratio = 1.0f;
                stats.rawBytes = static_cast<uint64_t>(ALIMAX(0.0, std::round(logicalRawBytes)));
                stats.compressedBytes = stats.rawBytes;
                stats.decompressedBytes = stats.rawBytes;
                stats.fallbackUsed = true;
                return stats;
            }
            const PredictorFlags keyFlags = parsePredictorFlags(mMeta->h2o_lossless_predictors_k, true);
            const PredictorFlags valueFlags = parsePredictorFlags(mMeta->h2o_lossless_predictors_v, true);
            const int flashBlockKv = ALIMAX(1, mKVCacheManager->getFlashAttentionBlockKv());
            const size_t keyBytesPerHead = (size_t)UP_DIV(evalTokens, hP) * (size_t)ROUND_UP(mHeadDim, lP) * (size_t)hP * (size_t)mBytes;
            const size_t valueBytesPerHead = (size_t)UP_DIV(evalTokens, flashBlockKv) * (size_t)ROUND_UP(mHeadDim, hP) * (size_t)ROUND_UP(flashBlockKv, lP) * (size_t)mBytes;
            uint64_t rawFullBytes = 0;
            uint64_t compressedFullBytes = 0;
            for (int h = 0; h < mKvNumHead; ++h) {
                const uint8_t* keyPtr = reinterpret_cast<const uint8_t*>(mKVCacheManager->addrOfKey(h));
                const uint8_t* valuePtr = reinterpret_cast<const uint8_t*>(mKVCacheManager->addrOfValue(h));
                const auto sample = encodeGearDeltaLosslessStats(
                    keyPtr,
                    keyBytesPerHead,
                    valuePtr,
                    valueBytesPerHead,
                    keyFlags,
                    valueFlags
                );
                rawFullBytes += sample.rawBytes;
                compressedFullBytes += sample.compressedBytes;
            }
            if (rawFullBytes == 0 || compressedFullBytes == 0) {
                stats.fallbackUsed = true;
                stats.rawBytes = static_cast<uint64_t>(ALIMAX(0.0, std::round(logicalRawBytes)));
                stats.compressedBytes = stats.rawBytes;
                stats.decompressedBytes = stats.rawBytes;
                return stats;
            }
            // If compression expands, count as runtime fallback and keep RAW.
            if (compressedFullBytes > rawFullBytes) {
                compressedFullBytes = rawFullBytes;
                stats.fallbackUsed = true;
            }
            stats.rawBytes = rawFullBytes;
            stats.compressedBytes = compressedFullBytes;
            stats.decompressedBytes = rawFullBytes;
            stats.ratio = compressedFullBytes > 0
                ? static_cast<float>((double)rawFullBytes / (double)compressedFullBytes)
                : 1.0f;
            return stats;
        };

        bool applyLossless = false;
        int losslessTokenBudget = kvSeqLen;
        if (mMeta->h2o_lossless_enable != 0
            && mMeta->h2o_lossless_codec == 1
            && mMeta->h2o_lossless_runtime_enable != 0) {
            if (mMeta->h2o_lossless_scope == 1) { // front_n
                applyLossless = layerIndex < ALIMAX(0, mMeta->h2o_lossless_front_n);
                losslessTokenBudget = kvSeqLen;
            } else if (mMeta->h2o_lossless_scope == 2) { // h2o_kept
                applyLossless = h2oLayerInRange;
                losslessTokenBudget = finalKeepTokensForLayer;
            }
        }
        if (applyLossless) {
            const int triggerMin = ALIMAX(1, mMeta->h2o_trigger_min_tokens);
            const int updateInterval = ALIMAX(1, mMeta->h2o_update_interval);
            const bool shouldUpdateLossless = kvSeqLen >= triggerMin
                && (mH2OState->globalStep - mH2OState->globalLastLosslessStep >= updateInterval);
            if (shouldUpdateLossless || mH2OState->globalLastLosslessRatio <= 1.0f) {
                auto c0 = std::chrono::high_resolution_clock::now();
                const auto stats = estimateLosslessRuntimeStats(losslessTokenBudget);
                auto c1 = std::chrono::high_resolution_clock::now();
                mH2OState->globalLastLosslessRatio = stats.ratio;
                mH2OState->globalLastLosslessCodecUs =
                    (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(c1 - c0).count();
                mH2OState->globalLastLosslessRawBytes = stats.rawBytes;
                mH2OState->globalLastLosslessCompressedBytes = stats.compressedBytes;
                mH2OState->globalLastLosslessDecompressedBytes = stats.decompressedBytes;
                mH2OState->globalLastLosslessCompressUs = mH2OState->globalLastLosslessCodecUs;
                mH2OState->globalLastLosslessDecompressUs = 0;
                mH2OState->globalLastLosslessStep = mH2OState->globalStep;
                if (stats.fallbackUsed) {
                    mH2OState->globalLosslessFallbackCount += 1;
                }
            }
        }
        // Always expose global lossless stats for current request.
        // front_n scope only updates on early layers, but bench reads context once per response.
        mMeta->h2o_lossless_ratio = mH2OState->globalLastLosslessRatio;
        mMeta->h2o_codec_us = mH2OState->globalLastLosslessCodecUs;
        mMeta->h2o_lossless_raw_bytes = mH2OState->globalLastLosslessRawBytes;
        mMeta->h2o_lossless_compressed_bytes = mH2OState->globalLastLosslessCompressedBytes;
        mMeta->h2o_lossless_decompressed_bytes = mH2OState->globalLastLosslessDecompressedBytes;
        mMeta->h2o_lossless_compress_us = mH2OState->globalLastLosslessCompressUs;
        mMeta->h2o_lossless_decompress_us = mH2OState->globalLastLosslessDecompressUs;
        mMeta->h2o_lossless_queue_depth_peak = mH2OState->globalLosslessQueueDepthPeak;
        mMeta->h2o_lossless_fallback_count = mH2OState->globalLosslessFallbackCount;
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
    mH2OState.reset(new H2OSharedState);
}

CPUAttention::~CPUAttention() {

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
