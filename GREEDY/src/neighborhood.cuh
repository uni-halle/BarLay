#ifndef GREEDY_NEIGHBORHOOD_CUH
#define GREEDY_NEIGHBORHOOD_CUH

#include "barcodes.cuh"
#include "layout.cuh"
#include "d_synth.cuh"

namespace neighborhood {
    using Neighborhood = barcodes::Schedule[4];

    __device__ __forceinline__ void load(
        const uint32_t x,
        const uint32_t y,
        const layout::Layout * __restrict__ layout,
        const barcodes::ScheduleSet * __restrict__ schedules,
        Neighborhood * __restrict__ neighbors
    ) {
        if (y > 0)
            (*neighbors)[0] = (*schedules)[layout->positions[x][y-1].i_schedule];

        if (x > 0 && y > 0)
            (*neighbors)[1] = (*schedules)[layout->positions[x-1][y-1].i_schedule];

        if (x > 0)
            (*neighbors)[2] = (*schedules)[layout->positions[x-1][y].i_schedule];

        if (x > 0 && y < COL_COUNT - 1)
            (*neighbors)[3] = (*schedules)[layout->positions[x-1][y+1].i_schedule];
    }

    __device__ __forceinline__ uint16_t nquality(
        const uint32_t x,
        const uint32_t y,
        const barcodes::Schedule * __restrict__ candidate,
        const barcodes::ScheduleSet * __restrict__ schedules,
        const Neighborhood * __restrict__ neighbors
    ) {
        uint16_t quality = 0;

        if (y > 0)
            quality += d_synth(candidate, &(*neighbors)[0]);

        if (x > 0 && y > 0)
            quality += d_synth(candidate, &(*neighbors)[1]);

        if (x > 0)
            quality += d_synth(candidate, &(*neighbors)[2]);

        if (x > 0 && y < COL_COUNT - 1)
            quality += d_synth(candidate, &(*neighbors)[3]);

        return quality;
    }
}
#endif //GREEDY_NEIGHBORHOOD_CUH