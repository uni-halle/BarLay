#ifndef GREEDY_LAYOUT_CUH
#define GREEDY_LAYOUT_CUH

#include "barcodes.cuh"

namespace layout {
    struct Position {
        uint32_t i_schedule;
    };

    struct Layout {
        Position positions[ROW_COUNT][COL_COUNT];
    };

    void print(FILE *target, const barcodes::ScheduleSet *schedules, const Layout *layout);
    void initialize(Layout *layout);
}

#endif //GREEDY_LAYOUT_CUH