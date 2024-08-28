#ifndef _LAYOUT_
#define _LAYOUT_

#include "params.hu"
#include "types.hu"

#include <stdint.h>
#include <stdio.h>

namespace layout {
    void print(FILE *target, const barcodes::ScheduleSet *schedules, const struct Layout *layout) {
        for (size_t x = 0; x < ROW_COUNT; x++) {
            for (size_t y = 0; y < COL_COUNT; y++) {
                const struct Position *position = &layout->positions[x][y];
                const barcodes::Schedule *schedule = &(*schedules)[position->i_schedule];
                // printf("position->i_schedule: %u\n", position->i_schedule);

                for (uint32_t i_bit = 0; i_bit < SCHEDULE_LENGTH; i_bit++) {
                    uint32_t i_chunk = i_bit / SYNTH_SCHEDULE_CHUNK_BIT_SIZE;
                    uint32_t chunk = schedule->chunks[i_chunk];
                    uint32_t i_chunk_bit = i_bit % SYNTH_SCHEDULE_CHUNK_BIT_SIZE;
                    uint32_t shifted_bit = uint32_t(1) << i_chunk_bit;

                    char bit = (chunk & shifted_bit) == 0 ? '0' : '1';

                    fputc(bit, target);
                }
                
                fputs("\n", target);
            }
        }
    }

    void initialize(struct Layout *layout) {
        uint32_t i = 0;
        for (size_t x = 0; x < ROW_COUNT; x++) {
            for (size_t y = 0; y < COL_COUNT; y++) {
                layout->positions[x][y].i_schedule = i;
                i++;
            }
        }
    }
}

#endif