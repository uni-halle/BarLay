#ifndef _BARCODES_
#define _BARCODES_

#include "params.hu"
#include "types.hu"

#include <stdint.h>
#include <stdio.h>

namespace barcodes {
    // calculate synthesis schedule (ACGT)
    void schedule(const struct Barcode *barcode, struct SynthSchedule *schedule) {
        size_t i_nucleotid = 0;
        for (size_t i_chunk = 0; i_chunk < BARCODE_LENGTH / 8 + 1; i_chunk++) {
            int chunk = 0;

            for (size_t i_cycle = 0; i_cycle < 8; i_cycle++) {
                if (i_nucleotid < BARCODE_LENGTH && barcode->nucleotides[i_nucleotid] == 'A') {
                    chunk += (1 << (4 * i_cycle));
                    i_nucleotid++;
                }

                if (i_nucleotid < BARCODE_LENGTH && barcode->nucleotides[i_nucleotid] == 'C') {
                    chunk += (1 << (4 * i_cycle + 1));
                    i_nucleotid++;
                }

                if (i_nucleotid < BARCODE_LENGTH && barcode->nucleotides[i_nucleotid] == 'G') {
                    chunk += (1 << (4 * i_cycle + 2));
                    i_nucleotid++;
                }

                if (i_nucleotid < BARCODE_LENGTH && barcode->nucleotides[i_nucleotid] == 'T') {
                    chunk += (1 << (4 * i_cycle + 3));
                    i_nucleotid++;
                }
            }

            schedule->chunks[i_chunk] = chunk;
        }
    }

    void read(FILE *source, Set *barcode_set, ScheduleSet *schedules) {
        for (size_t i_barcode = 0; i_barcode < NUMBER_OF_BARCODES; i_barcode++) {
            struct Barcode *barcode = &(*barcode_set)[i_barcode];

            for (size_t i_nucleotid = 0; i_nucleotid < BARCODE_LENGTH; i_nucleotid++)
                barcode->nucleotides[i_nucleotid] = fgetc(source); // read nucleotides
            
            fgetc(source); // read newline

            schedule(barcode, &(*schedules)[i_barcode]);
        }
    }

    __device__ uint16_t d_synth(const barcodes::SynthSchedule *s1, const barcodes::SynthSchedule *s2) {
        int distance = 0;

        for (size_t i_chunk = 0; i_chunk < SYNTH_SCHEDULE_CHUNKS; i_chunk++)
            distance += __builtin_popcount(s1->chunks[i_chunk] ^ s2->chunks[i_chunk]);

        return distance;
    }
}

#endif