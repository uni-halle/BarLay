#ifndef GREEDY_INDEX_POOL_CUH
#define GREEDY_INDEX_POOL_CUH

#include "config.cuh"

namespace index_pool {
    struct Pool {
        uint32_t size;
        uint32_t indices[NUMBER_OF_SCHEDULES];
    };

    __global__ void prepare(struct Pool *pool);
    __device__ uint32_t poll(struct Pool *pool, uint32_t index);
}

#endif //GREEDY_INDEX_POOL_CUH