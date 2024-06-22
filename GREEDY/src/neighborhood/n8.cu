#include "../params.hu"
#include "../types.hu"
#include "../d_synth.cu"

namespace neighborhood {

    using Neighborhood = barcodes::SynthSchedule[4];

    __device__ __forceinline__ void load(
        uint32_t x,
        uint32_t y,
        layout::Layout *layout,
        barcodes::ScheduleSet *schedules,
        Neighborhood* neighbors
    ) {
        if (y > 0)
            (*neighbors)[0] = (*schedules)[layout->positions[x][y-1].i_barcode];

        if (x > 0 && y > 0)
            (*neighbors)[1] = (*schedules)[layout->positions[x-1][y-1].i_barcode];

        if (x > 0)
            (*neighbors)[2] = (*schedules)[layout->positions[x-1][y].i_barcode];

        if (x > 0 && y < DIM_Y - 1)
            (*neighbors)[3] = (*schedules)[layout->positions[x-1][y+1].i_barcode];
    }

    __device__ __forceinline__ uint16_t nquality(
        uint32_t x,
        uint32_t y,
        barcodes::SynthSchedule *candidate,
        barcodes::ScheduleSet *schedules,
        Neighborhood* neighbors
    ) {
        uint16_t quality = 0;

        if (y > 0)
            quality += d_synth(candidate, &(*neighbors)[0]);

        if (x > 0 && y > 0)
            quality += d_synth(candidate, &(*neighbors)[1]);

        if (x > 0)
            quality += d_synth(candidate, &(*neighbors)[2]);

        if (x > 0 && y < DIM_Y - 1)
            quality += d_synth(candidate, &(*neighbors)[3]);
        
       return quality;
    }
}