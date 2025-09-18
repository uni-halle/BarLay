#ifndef GREEDY_BARCODES_CUH
#define GREEDY_BARCODES_CUH

#include "config.cuh"

#include <cstdint>
#include <cstdio>

namespace barcodes {
    struct Schedule {
        uint32_t chunks[SYNTH_SCHEDULE_CHUNKS];
    };

    using ScheduleSet = Schedule[NUMBER_OF_SCHEDULES];

    void read(FILE *source, ScheduleSet *schedules);
}

#endif //GREEDY_BARCODES_CUH