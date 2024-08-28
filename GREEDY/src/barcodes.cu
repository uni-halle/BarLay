#ifndef _BARCODES_
#define _BARCODES_

#include "params.hu"
#include "types.hu"

#include <stdint.h>
#include <stdio.h>

namespace barcodes {
    void read(FILE *source, ScheduleSet *schedules) {
        for (size_t i_schedule = 0; i_schedule < NUMBER_OF_SCHEDULES; i_schedule++) {
            struct Schedule *schedule = &(*schedules)[i_schedule];

            
            for (size_t i_chunk = 0; i_chunk < SYNTH_SCHEDULE_CHUNKS; i_chunk++) {
                schedule->chunks[i_chunk] = 0;
            }

            for (size_t i_bit = 0; i_bit < SCHEDULE_LENGTH; i_bit++) {
                char bit = fgetc(source); // read bits

                uint32_t i_chunk = i_bit / SYNTH_SCHEDULE_CHUNK_BIT_SIZE;
                uint32_t i_chunk_bit = i_bit % SYNTH_SCHEDULE_CHUNK_BIT_SIZE;
                uint32_t shifted_bit = uint32_t(1) << i_chunk_bit;
                uint32_t conditional_or_operand = (bit == '1' ? shifted_bit : 0);

                schedule->chunks[i_chunk] |= conditional_or_operand;
            }
            
            fgetc(source); // read newline
        }
    }

    __device__ uint16_t d_synth(const barcodes::Schedule *s1, const barcodes::Schedule *s2) {
        int distance = 0;

        for (size_t i_chunk = 0; i_chunk < SYNTH_SCHEDULE_CHUNKS; i_chunk++)
            distance += __popc(s1->chunks[i_chunk] ^ s2->chunks[i_chunk]);

        return distance;
    }
}

#endif