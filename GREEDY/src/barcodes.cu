#include "config.cuh"
#include "barcodes.cuh"

#include <cstdio>

namespace barcodes {
    void read(FILE *source, ScheduleSet *schedules) {
        for (size_t i_schedule = 0; i_schedule < NUMBER_OF_SCHEDULES; i_schedule++) {
            Schedule *schedule = &(*schedules)[i_schedule];
            
            for (size_t i_chunk = 0; i_chunk < SYNTH_SCHEDULE_CHUNKS; i_chunk++) {
                schedule->chunks[i_chunk] = 0;
            }

            for (size_t i_bit = 0; i_bit < SCHEDULE_LENGTH; i_bit++) {
                const int bit = fgetc(source); // read bits

                uint32_t i_chunk = i_bit / SYNTH_SCHEDULE_CHUNK_BIT_SIZE;
                uint32_t i_chunk_bit = i_bit % SYNTH_SCHEDULE_CHUNK_BIT_SIZE;
                uint32_t shifted_bit = static_cast<uint32_t>(1) << i_chunk_bit;
                uint32_t conditional_or_operand = (bit == '1' ? shifted_bit : 0);

                schedule->chunks[i_chunk] |= conditional_or_operand;
            }
            
            fgetc(source); // read newline
        }
    }
}