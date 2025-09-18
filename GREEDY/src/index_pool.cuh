#ifndef GREEDY_INDEX_POOL_CUH
#define GREEDY_INDEX_POOL_CUH

#include "config.cuh"

namespace index_pool {
    struct Pool {
        uint32_t size;
        uint32_t indices[NUMBER_OF_SCHEDULES];
    };

    __global__ void prepare(Pool * __restrict__ pool);
    __device__ uint32_t poll(Pool * __restrict__ pool, uint32_t index);
}

#endif //GREEDY_INDEX_POOL_CUH