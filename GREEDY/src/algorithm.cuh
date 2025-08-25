#ifndef GREEDY_ALGORITHM_CUH
#define GREEDY_ALGORITHM_CUH

#include "index_pool.cuh"

#define THREAD_BLOCKS 1024
#define THREADS_PER_THREAD_BLOCK 256

namespace algorithm {
    struct BarcodeQuality {
        uint16_t quality;
        uint32_t indexPoolHandle;
    };

    using ResultBlock = struct BarcodeQuality[THREAD_BLOCKS];

    __global__ void grid_reduction(
        uint32_t x,
        uint32_t y,
        const layout::Layout *layout,
        const barcodes::ScheduleSet *schedules,
        const index_pool::Pool *pool,
        ResultBlock *result_block
    );

    __global__ void block_reduction(
        size_t x,
        size_t y,
        layout::Layout *layout,
        index_pool::Pool *pool,
        const ResultBlock *result_block
    );

    void optimize(const barcodes::ScheduleSet *schedules, layout::Layout *result);
}

#endif //GREEDY_ALGORITHM_CUH