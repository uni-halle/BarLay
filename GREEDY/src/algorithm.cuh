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

    using ResultBlock = BarcodeQuality[THREAD_BLOCKS];

    __global__ void grid_reduction(
        uint32_t x,
        uint32_t y,
        const layout::Layout * __restrict__ layout,
        const barcodes::ScheduleSet * __restrict__ schedules,
        const index_pool::Pool * __restrict__ pool,
        ResultBlock * __restrict__ result_block
    );

    __global__ void block_reduction(
        size_t x,
        size_t y,
        layout::Layout * __restrict__ layout,
        index_pool::Pool * __restrict__ pool,
        const ResultBlock * __restrict__ result_block
    );

    void optimize(const barcodes::ScheduleSet * __restrict__ schedules, layout::Layout * __restrict__ result);
}

#endif //GREEDY_ALGORITHM_CUH