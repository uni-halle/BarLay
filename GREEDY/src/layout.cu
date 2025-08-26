#include "config.cuh"
#include "layout.cuh"

#include <cstdint>
#include <cstdio>

#include "occupancy.cuh"

namespace layout {
    void print(FILE *target, const barcodes::ScheduleSet *schedules, const Layout *layout) {
        for (uint16_t x = 0; x < ROW_COUNT; x++) {
            for (uint16_t y = 0; y < COL_COUNT; y++) {
                if (occupancy::should_be_filled(x,y)) {
                    const Position *position = &layout->positions[x][y];
                    const barcodes::Schedule *schedule = &(*schedules)[position->i_schedule];

                    for (uint32_t i_bit = 0; i_bit < SCHEDULE_LENGTH; i_bit++) {
                        const uint32_t i_chunk = i_bit / SYNTH_SCHEDULE_CHUNK_BIT_SIZE;
                        const uint32_t chunk = schedule->chunks[i_chunk];
                        const uint32_t i_chunk_bit = i_bit % SYNTH_SCHEDULE_CHUNK_BIT_SIZE;
                        const uint32_t shifted_bit = static_cast<uint32_t>(1) << i_chunk_bit;

                        const char bit = (chunk & shifted_bit) == 0 ? '0' : '1';

                        fputc(bit, target);
                    }
                }

                fputs("\n", target);
            }
        }
    }

    void initialize(Layout *layout) {
        uint32_t i = 0;
        for (size_t x = 0; x < ROW_COUNT; x++) {
            for (size_t y = 0; y < COL_COUNT; y++) {
                layout->positions[x][y].i_schedule = i;
                i++;
            }
        }
    }
}
