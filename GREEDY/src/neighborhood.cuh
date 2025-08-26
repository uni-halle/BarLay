#ifndef GREEDY_NEIGHBORHOOD_CUH
#define GREEDY_NEIGHBORHOOD_CUH

#include "barcodes.cuh"
#include "layout.cuh"
#include "helpers.cuh"
#include "occupancy.cuh"

namespace neighborhood {
    using Neighborhood = barcodes::Schedule[4];

    __device__ __forceinline__ void load(
        const uint16_t x,
        const uint16_t y,
        const layout::Layout * __restrict__ layout,
        const barcodes::ScheduleSet * __restrict__ schedules,
        Neighborhood * __restrict__ neighbors
    ) {
        #if NEIGHBORHOOD == NEIGHBORHOOD_N8
        if (occupancy::should_be_filled(x, y-1))
            (*neighbors)[0] = (*schedules)[layout->positions[x][y-1].i_schedule];

        if (occupancy::should_be_filled(x-1, y-1))
            (*neighbors)[1] = (*schedules)[layout->positions[x-1][y-1].i_schedule];

        if (occupancy::should_be_filled(x-1, y))
            (*neighbors)[2] = (*schedules)[layout->positions[x-1][y].i_schedule];

        if (occupancy::should_be_filled(x-1, y+1))
            (*neighbors)[3] = (*schedules)[layout->positions[x-1][y+1].i_schedule];
        #endif
    }

    __device__ __forceinline__ uint16_t nquality(
        const uint16_t x,
        const uint16_t y,
        const barcodes::Schedule * __restrict__ candidate,
        const Neighborhood * __restrict__ neighbors
    ) {
        #if NEIGHBORHOOD == NEIGHBORHOOD_N8
        uint16_t quality = 0;

        if (occupancy::should_be_filled(x, y-1))
            quality += helpers::d_synth(candidate, &(*neighbors)[0]);

        if (occupancy::should_be_filled(x-1, y-1))
            quality += helpers::d_synth(candidate, &(*neighbors)[1]);

        if (occupancy::should_be_filled(x-1, y))
            quality += helpers::d_synth(candidate, &(*neighbors)[2]);

        if (occupancy::should_be_filled(x-1, y+1))
            quality += helpers::d_synth(candidate, &(*neighbors)[3]);

        return quality;
        #endif
    }
}
#endif //GREEDY_NEIGHBORHOOD_CUH