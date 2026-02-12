//
//  CPUAttention.hpp
//  MNN
//
//  Created by MNN on 2024/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef CPUATTENTION_HPP
#define CPUATTENTION_HPP

#include <memory>
#include <vector>
#include <functional>
#include <cstdint>
#include "core/Execution.hpp"
#include "core/OpCommonUtils.hpp"
#include "CPUKVCacheManager.hpp"
#include "MNN/ErrorCode.hpp"

namespace MNN {

class CPUAttention : public Execution {
public:
    CPUAttention(Backend *backend, bool kv_cache);
    virtual ~CPUAttention();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    struct H2OSharedState {
        struct LayerState {
            std::vector<float> blockScores;
        };
        std::vector<LayerState> layerStates;
        std::vector<int> reserveStorage;
        int64_t decodeLayerCursor = 0;
        int64_t globalStep = 0;
        int64_t globalLastTriggerStep = 0;
        int64_t globalLastLosslessStep = 0;
        float globalLastLosslessRatio = 1.0f;
        int64_t globalLastLosslessCodecUs = 0;
        uint64_t globalLastLosslessRawBytes = 0;
        uint64_t globalLastLosslessCompressedBytes = 0;
        uint64_t globalLastLosslessDecompressedBytes = 0;
        int64_t globalLastLosslessCompressUs = 0;
        int64_t globalLastLosslessDecompressUs = 0;
        int64_t globalLosslessQueueDepthPeak = 0;
        int64_t globalLosslessFallbackCount = 0;
    };

    bool mKVCache        = true;
    int mBytes = 4;
    int mThreadNum = 1;
    int mBlockKV = 512;
    int eP, lP, hP, mPack; // float matmul packing
    int eP8, lP8, hP8;    // GemmInt8 packing
    int mNumHead, mKvNumHead, mHeadDim;
    KVMeta* mMeta;

    // common
    std::shared_ptr<Tensor> mPackQ, mPackQKV, mRunningMax, mRunningSum, mTempQKBlock, mTempOut, mExpfDiffMax;
    std::shared_ptr<CPUKVCacheManager> mKVCacheManager = nullptr;
    bool mUseFlashAttention = true;

    // quant Query/Key/Value
    bool mQuantKey   = false;
    bool mQuantValue = false;
    int  mBlockNum   = 1;
    MemChunk mSumQ;
    MemChunk mQueryScale, mQueryZeroPoint, mQueryQuantScale, mQueryQuantZero;
    MemChunk mQuantQuery, mAccumBuffer;

    MemChunk mQuantQK, mQKScale, mQKBias, mSumQK, mArray;
    AutoStorage<int8_t> mGemmBias, mGemmRelu;
    std::shared_ptr<H2OSharedState> mH2OState;

    std::function<void(const float*, int8_t*, size_t, const float*, ssize_t, ssize_t, const float*, ssize_t)> mQuantFunc;
    decltype(CoreInt8Functions::Int8GemmKernel) mInt8GemmKernel;
};

} // namespace MNN

#endif // CPUATTENTION_HPP

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
