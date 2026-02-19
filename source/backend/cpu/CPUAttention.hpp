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
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
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
            struct LosslessBlock {
                enum Origin : uint8_t {
                    FROM_RAW_KV = 0,
                    FROM_DROPPED_KV = 1,
                };
                int startToken = 0;
                int tokenCount = 0;
                uint64_t rawBytes = 0;
                uint64_t compressedBytes = 0;
                uint64_t rawHash = 0;
                uint64_t blobHash = 0;
                bool decodedOnce = false;
                bool rawDropped = false;
                Origin origin = FROM_RAW_KV;
                std::vector<uint8_t> keyBlob;
                std::vector<uint8_t> valueBlob;
            };
            struct DecodedCacheEntry {
                int startToken = 0;
                int tokenCount = 0;
                uint64_t rawHash = 0;
                uint64_t blobHash = 0;
                uint64_t rawBytes = 0;
                uint64_t compressedBytes = 0;
                std::vector<uint8_t> keyDecoded;
                std::vector<uint8_t> valueDecoded;
            };
            std::vector<float> blockScores;
            // Per-layer pending compact plan for next decode step.
            // Kept in layer-local storage to avoid cross-layer pointer aliasing.
            size_t pendingRemove = 0;
            std::vector<int> activeReservePairs;
            std::vector<int> pendingReservePairs;
            int pendingPlanReady = 0;
            int64_t losslessLastStep = 0;
            int losslessLastTokenBudget = 0;
            uint64_t losslessRawBytes = 0;
            uint64_t losslessCompressedBytes = 0;
            uint64_t losslessDecompressedBytes = 0;
            int64_t losslessCompressUs = 0;
            int64_t losslessDecompressUs = 0;
            int64_t losslessFallbackCount = 0;
            int64_t losslessUpdateCount = 0;
            int64_t losslessBackpressureSkipCount = 0;
            std::vector<LosslessBlock> losslessBlocks;
            std::deque<DecodedCacheEntry> decodeCacheEntries;
        };
        struct AsyncResult {
            uint64_t taskId = 0;
            int layerIndex = 0;
            int startToken = 0;
            int tokenCount = 0;
            bool runtimeStoreMode = false;
            bool fallbackUsed = false;
            bool backpressureSkipped = false;
            uint64_t rawBytes = 0;
            uint64_t compressedBytes = 0;
            uint64_t decompressedBytes = 0;
            uint64_t attemptedCompressedBytes = 0;
            uint64_t rawHash = 0;
            uint64_t blobHash = 0;
            float attemptedRatio = 1.0f;
            float ratio = 1.0f;
            int64_t compressUs = 0;
            int64_t decompressUs = 0;
            std::vector<uint8_t> keyBlob;
            std::vector<uint8_t> valueBlob;
        };
        struct AsyncTask {
            uint64_t taskId = 0;
            int layerIndex = 0;
            int startToken = 0;
            int tokenCount = 0;
            bool runtimeStoreMode = false;
            std::function<AsyncResult()> fn;
        };
        std::vector<LayerState> layerStates;
        // reserveStorage: active compact plan used by meta->reserve in current step.
        // reserveStoragePending: next-step plan generated during this step.
        std::vector<int> reserveStorage;
        std::vector<int> reserveStoragePending;
        // Cross-layer decoded cache to improve reuse for repeated compressed blocks.
        std::deque<LayerState::DecodedCacheEntry> globalDecodeCacheEntries;
        std::mutex asyncMutex;
        std::condition_variable asyncCv;
        std::condition_variable asyncDoneCv;
        std::deque<AsyncTask> asyncTasks;
        std::deque<AsyncResult> asyncCompleted;
        std::vector<std::thread> asyncWorkers;
        bool asyncStop = false;
        int asyncConfiguredThreads = 0;
        int64_t asyncPendingTasks = 0;
        int64_t asyncRunningTasks = 0;
        uint64_t asyncTaskSerial = 0;
        int64_t decodeLayerCursor = 0;
        int64_t globalStep = 0;
        int64_t globalTokenStep = 0;
        int64_t globalLastTriggerStep = 0;
        int64_t globalLastLosslessStep = 0;
        int globalLastLosslessTokenBudget = 0;
        float globalLastLosslessRatio = 1.0f;
        int64_t globalLastLosslessCodecUs = 0;
        uint64_t globalLastLosslessRawBytes = 0;
        uint64_t globalLastLosslessCompressedBytes = 0;
        uint64_t globalLastLosslessDecompressedBytes = 0;
        int64_t globalLastLosslessCompressUs = 0;
        int64_t globalLastLosslessDecompressUs = 0;
        int64_t globalLosslessQueueDepthPeak = 0;
        int64_t globalLosslessFallbackCount = 0;
        int64_t globalLosslessBackpressureSkipCount = 0;
        int64_t globalLosslessAsyncQueuePeak = 0;
        int64_t globalLosslessAsyncWaitUs = 0;
        int64_t globalLosslessDecodeCacheHit = 0;
        int64_t globalLosslessDecodeCacheMiss = 0;
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
