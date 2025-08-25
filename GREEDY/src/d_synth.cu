#include "config.cuh"
#include "d_synth.cuh"
#include "barcodes.cuh"

__device__ int d_synth(barcodes::Schedule *s1, barcodes::Schedule *s2)
{
    int distance = 0;

    for (size_t i_chunk = 0; i_chunk < SYNTH_SCHEDULE_CHUNKS; i_chunk++)
        distance += __popc(s1->chunks[i_chunk] ^ s2->chunks[i_chunk]);

    return distance;
}