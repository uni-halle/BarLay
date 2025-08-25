#include "config.cuh"
#include "index_pool.cuh"

namespace index_pool {
    __global__ void prepare(struct Pool * __restrict__ pool) {
        pool->size = NUMBER_OF_SCHEDULES;
        for (size_t i = 0; i < NUMBER_OF_SCHEDULES; i++)
            pool->indices[i] = i;
    }

    __device__ uint32_t poll(struct Pool * __restrict__ pool, const uint32_t index) {
        // save polled index
        const uint32_t polled = pool->indices[index];

        // replace polled index with index in last position
        pool->indices[index] = pool->indices[pool->size - 1];

        // reduce pool size
        pool->size--;

        return polled;
    }
}
