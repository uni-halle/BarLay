#ifndef GREEDY_OCCUPANCY_CUH
#define GREEDY_OCCUPANCY_CUH
#include "helpers.cuh"

namespace occupancy {
    __device__ __host__ __forceinline__ bool should_be_filled(const uint16_t x, const uint16_t y) {
        #if (OCCUPANCY == OCCUPANCY_FULL)
        return helpers::index_in_range(x, 0, ROW_COUNT) && helpers::index_in_range(y, 0, COL_COUNT);
        #elif (OCCUPANCY == OCCUPANCY_CHESSBOARD)
        return helpers::index_in_range(x, 0, ROW_COUNT)
            && helpers::index_in_range(y, 0, COL_COUNT)
            && (x + y) % 2;
        #endif
    }
}
#endif //GREEDY_OCCUPANCY_CUH